import math
import torch
import torch.nn as nn
from ..registry import TOP_MODEL
from ..utils import normal_init
from .top_model import TopModel
import threading

@TOP_MODEL.register_module
class AmSoftmax1DeviceMTModel(TopModel):
    def __init__(self, feature_dim, num_classes, m=0.35, s=30):
        gpus = list(range(0, torch.cuda.device_count() - 3))
        super(AmSoftmax1DeviceMTModel, self).__init__(feature_dim, num_classes, gpus)
        self.m = m
        self.s = s
        self.device_count = torch.cuda.device_count()
        self.ws = nn.ParameterList([])
        self.num_classes = num_classes
        # self.probs = torch.zeros([batch_size, num_classes], device='cuda:0')
        for i in range(self.device_count - 3):
            self.ws.append(torch.nn.Parameter(
                torch.randn(feature_dim, self.classes_nums[i], requires_grad=True, device='cuda:{}'.format(i))))
            # self.ces.append(nn.CrossEntropyLoss().to('cuda:{}'.format(i)))
        self.ce = nn.DataParallel(nn.CrossEntropyLoss().to('cuda:6'), device_ids=[6,7], output_device=6)
        self.init_weights()

    def init_weights(self, pretrained=None):
        for w in self.ws:
            w.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def _forward_train(self, input, label):
        self.probs = torch.zeros([input.shape[0], self.num_classes], device='cuda:5')
        print('_forward_train')
        feat_norm = torch.norm(input, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        normed_feat = input / feat_norm
        device_first = 'cuda:0'
        # self.probs = torch.zeros([input.shape[0], self.num_classes], device='cuda:0')
        lock = threading.Lock()
        def set_cosine_m_s(w, i):
            device_i = 'cuda:{}'.format(i)
            label_i = label.to(device_i)
            drop_idxes = ((label_i < self.classes_parts[i][0]) | (label_i >= self.classes_parts[i][1]))
            label_i[drop_idxes] = self.classes_parts[i][1]
            label_i -= self.classes_parts[i][0]
            label_i_view = label_i.view(-1, 1)
            normed_feat_i = normed_feat.to(device_i)
            w_norm = torch.norm(w, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            normed_w = w / w_norm
            cosine = torch.mm(normed_feat_i, normed_w)
            # from insightface_pytorch, for numerical stability
            cosine = cosine.clamp(-1, 1)
            m = torch.zeros((cosine.size()[0], cosine.size()[1] + 1), device=device_i).scatter_(1, label_i_view, self.m)
            cosine_m_s = self.s * (cosine - m[:, :-1])
            with lock:
                self.probs[:, self.classes_parts[i][0]:self.classes_parts[i][1]] = cosine_m_s
                # cosine_m_ss[i - 1] = cosine_m_s.to(device_first)
        ts = []
        for i, w in enumerate(self.ws):
            t = threading.Thread(target=set_cosine_m_s, args=(w, i))
            t.start()
            ts.append(t)
        for t in ts:
            t.join()
        # total_cosine_m_s = torch.cat(cosine_m_ss, dim=1)
        # loss = self.ce(total_cosine_m_s, label)
        loss = self.ce(self.probs, label)
        return loss

    def _forward_val(self, input, label):
        input = input.to(torch.device('cuda:0'))
        label = label.to(torch.device('cuda:0'))
        es_dict = self._forward(input, label)
        es = es_dict['es']

        preds = []
        idxes = []
        for i in range(len(es)):
            pred, idx = es[i].max(dim=1)
            preds.append(pred.view((-1, 1)).to('cuda:0'))
            idxes.append((idx.view((-1, 1)) + self.classes_parts[i][0]).to('cuda:0'))
        preds = torch.cat(preds, dim=1)
        idxes = torch.cat(idxes, dim=1)
        _, _idx = torch.max(preds, dim=1)
        pred_idxes = idxes[list(range(self.batch_size)), _idx]
        acc = sum((pred_idxes == label).float()) / len(label)
        return acc

    def forward(self, input, label, mode='train'):
        if mode == 'train':
            return self._forward_train(input, label)
        elif mode == 'val':
            return self._forward_val(input, label)


