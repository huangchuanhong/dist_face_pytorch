import math
import torch 
import torch.nn as nn
import threading
from ..registry import TOP_MODEL
from ..utils import normal_init

class TopModel(nn.Module):
    def __init__(self, feature_dim, num_classes, gpus=None, **kwargs):
        super(TopModel, self).__init__()
        if gpus is None:
            self.gpus = list(range(torch.cuda.device_count()))
        else:
            self.gpus = gpus
        self.classes_nums = []
        self.classes_parts = []
        num_classes_per_gpu = math.ceil(num_classes / len(self.gpus))
        cur_class = 0
        for i in range(torch.cuda.device_count()):#len(self.gpus)):
            if i not in self.gpus:
                self.classes_nums.append(0)
                self.classes_parts.append([cur_class, cur_class])
            elif i != self.gpus[-1]:
                self.classes_nums.append(num_classes_per_gpu)
                self.classes_parts.append([cur_class, cur_class + num_classes_per_gpu])
                cur_class += num_classes_per_gpu
            else:
                self.classes_nums.append(num_classes - num_classes_per_gpu * (len(self.gpus) - 1))
                self.classes_parts.append([cur_class, num_classes])
                cur_class = num_classes

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
        label_i_view = label_i.view(-1, 1)
        m = torch.zeros((cosine.size()[0], cosine.size()[1] + 1), device=device_i).scatter_(1, label_i_view, self.m)
        cosine_m_s = self.s * (cosine - m[:, :-1])
        return cosine_m_s

    def _forward_train(self, input, label):
        feat_norm = torch.norm(input, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        normed_feat = input / feat_norm
        device_first = 'cuda:0'
        logsumexps = [None] * len(self.ws)
        x = torch.zeros([input.shape[0]], device='cuda:0')
        # xs = [None] * len(self.ws)
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
            # m = torch.zeros((cosine.size()[0], cosine.size()[1] + 1), device=device_i).scatter_(1, label_i_view, self.m)
            # cosine_m_s = self.s * (cosine - m[:, :-1])
            # get labeled value from cosine_m_s
            label_i[drop_idxes] = 0
            # x_i = torch.gather(cosine_m_s, 1, label_i.view(-1, 1)) * (1 - drop_idxes.float().view((-1, 1)))
            x_i = torch.gather(probs_i, 1, label_i.view(-1, 1)).view(-1)
            with lock:
                x[1 - drop_idxes] = x_i[1 - drop_idxes].to(device_first)
                # xs[i] = x_i.to(device_first)
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
        total_logsumexp = torch.logsumexp(torch.cat(logsumexps, dim=1), 1)
        # print('total_logsumexp={}'.format(total_logsumexp))
        # loss = (total_logsumexp - torch.cat(xs, dim=1).sum(dim=1)).mean()
        loss = (total_logsumexp - x).mean()
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

