import math
import torch 
import torch.nn as nn
import threading
from collections import defaultdict
import torch.distributed as dist
from ..registry import TOP_MODEL
from ..utils import normal_init

class NdeviceTopModel(nn.Module):
    def __init__(self, batch_size, feature_dim, num_classes, top_model_ranks, top_model_gpus, top_group, **kwargs):
        super(NdeviceTopModel, self).__init__()
        self.batch_size = batch_size
        self.gpus = top_model_gpus
        self.top_model_ranks = top_model_ranks
        num_classes_per_rank = math.ceil(num_classes / len(self.top_model_ranks))
        self.total_classes_nums = defaultdict(list)
        self.total_classes_parts = defaultdict(list)
        for i, rank in enumerate(self.top_model_ranks):
            rank_start = num_classes_per_rank * i
            rank_end = min(num_classes_per_rank * (i + 1), num_classes)
            num_classes_per_gpu = math.ceil((rank_end - rank_start) / len(self.gpus))
            cur_class = rank_start
            for i in range(torch.cuda.device_count()):
                if i not in self.gpus:
                    self.total_classes_nums[rank].append(0)
                    self.total_classes_parts[rank].append([cur_class, cur_class])
                elif i != self.gpus[-1]:
                    self.total_classes_nums[rank].append(num_classes_per_gpu)
                    self.total_classes_parts[rank].append([cur_class, cur_class + num_classes_per_gpu])
                    cur_class += num_classes_per_gpu
                else:
                    self.total_classes_nums[rank].append(rank_end - cur_class)
                    self.total_classes_parts[rank].append([cur_class, rank_end])
                    cur_class = rank_end
        self.rank = dist.get_rank()
        self.classes_nums = self.total_classes_nums[self.rank]
        self.classes_parts = self.total_classes_parts[self.rank]
        self.ws = nn.ParameterList([])
        for i in range(torch.cuda.device_count()):
            self.ws.append(torch.nn.Parameter(
                torch.randn(feature_dim, self.classes_nums[i], requires_grad=True, device='cuda:{}'.format(i))))
        self.init_weights()
        self.top_group = top_group
        self.total_logsumexp_list = []
        self.total_x_list = []
        # self.total_x = torch.zeros((self.batch_size,), dtype=torch.float32, device='cuda:0')
        for i, rank in enumerate(self.top_model_ranks):
            if rank == self.rank:
                self.total_logsumexp_list.append(None)
                self.total_x_list.append(None)
            else:
                self.total_logsumexp_list.append(torch.zeros((self.batch_size, 1), dtype=torch.float32, device='cuda:0'))
                self.total_x_list.append(torch.zeros((self.batch_size,), dtype=torch.float32, device='cuda:0'))

    def get_probs_i(self, cosine, label_i, device_i):
        '''
        :param cosine: cosine similarity, shape:(batch_size, self.classes_nums[i])
        :param label_i: (1) self.classes_parts[i][0] <= origin_label < self.classes_parts[i][1]:
                                label = origin_label - self.classes_parts[i][0]
                        (2) else:
                                label = self.classes_parts[i][1]
        :param device_i: which gpu
        :return: probs
        '''
        # label_i_view = label_i.view(-1, 1)
        # m = torch.zeros((cosine.size()[0], cosine.size()[1] + 1), device=device_i).scatter_(1, label_i_view, self.m)
        # cosine_m_s = self.s * (cosine - m[:, :-1])
        # return cosine_m_s
        pass

    def _forward_train(self, input, label):
        feat_norm = torch.norm(input, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        normed_feat = input / feat_norm
        device_first = 'cuda:0'
        logsumexps = [None] * len(self.ws)
        # xs = [None] * len(self.ws)
        total_x = torch.zeros((self.batch_size,), dtype=torch.float32, device='cuda:0')
        lock = threading.Lock()
        def set_logsumexp(w, i):
            device_i = 'cuda:{}'.format(i)
            if i == 0:
                label_i = label.clone()
            else:
                label_i = label.to(device_i)
            drop_idxes = ((label_i < self.classes_parts[i][0]) | (label_i >= self.classes_parts[i][1]))
            # get (cosine - m) * s
            label_i[drop_idxes] = self.classes_parts[i][1]
            label_i -= self.classes_parts[i][0]
            # label_i_view = label_i.view(-1, 1)
            normed_feat_i = normed_feat.to(device_i)
            w_norm = torch.norm(w, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            normed_w = w / w_norm
            cosine = torch.mm(normed_feat_i, normed_w)
            # from insightface_pytorch, for numerical stability
            cosine = cosine.clamp(-1, 1)
            probs_i = self.get_probs_i(cosine, label_i, device_i)
            label_i[drop_idxes] = 0
            x_i = torch.gather(probs_i, 1, label_i.view(-1, 1)).view(-1)
            with lock:
                total_x[1 - drop_idxes] = x_i[1 - drop_idxes].to(device_first)
            # logsumexp
            logsumexp = torch.logsumexp(probs_i, 1, keepdim=True)
            with lock:
                logsumexps[i] = logsumexp.to(device_first)
        ts = []
        for i, w in enumerate(self.ws):
            t = threading.Thread(target=set_logsumexp, args=(w, i))
            t.start()
            ts.append(t)
        for t in ts:
            t.join()
        rank_logsumexp = torch.logsumexp(torch.cat(logsumexps, dim=1), 1, keepdim=True)
        for rank in self.top_model_ranks:
            if rank == self.rank:
                self.total_logsumexp_list[rank] = rank_logsumexp
                self.total_x_list[rank] = total_x
            dist.broadcast(tensor=self.total_logsumexp_list[rank], src=rank, group=self.top_group)
            dist.barrier(group=self.top_group)
            dist.broadcast(tensor=self.total_x_list[rank], src=rank, group=self.top_group)
            dist.barrier(group=self.top_group)
        total_logsumexp = torch.logsumexp(torch.cat(self.total_logsumexp_list, dim=1), 1)
        total_x = sum(self.total_x_list)
        loss = (total_logsumexp - total_x).mean()
        return loss

    def init_weights(self, pretrained=None):
        pass

    def forward(self, input, label, mode='train'):
        if mode == 'train':
            return self._forward_train(input, label)
        elif mode == 'val':
            return self._forward_val(input, label)

    def _forward_val(self, input, label):
        pass

