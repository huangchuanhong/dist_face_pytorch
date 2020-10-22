import math
import torch 
import torch.nn as nn
from ..registry import TOP_MODEL
from ..utils import normal_init
from .top_model import TopModel

import time
import logging
_format = '%(asctime)s - %(levelname)s - %(message)s'

logging.basicConfig(
    format=_format, level='INFO'
)
_logger = logging.getLogger(__name__)


@TOP_MODEL.register_module
class AmSoftmax1GPUModel(nn.Module):
    def __init__(self, feature_dim, num_classes, m=0.35, s=30, bs_per_gpu=200, gpus=[5,6]):
        super(AmSoftmax1GPUModel, self).__init__()
        self.m = m
        self.s = s
        self.device = torch.cuda.current_device()
        self.w = torch.nn.Parameter(torch.randn(feature_dim, num_classes, requires_grad=True))
        self.init_weights()
        self.ce = nn.CrossEntropyLoss()
        self.gpu_tmp_map = {}
        for i in gpus:
            self.gpu_tmp_map[i] = -self.m * torch.ones((bs_per_gpu, num_classes), device='cuda:{}'.format(i))
  
    def init_weights(self, pretrained=None):
        self.w.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
    
    def _forward_train(self, input, label):
        feat_norm = torch.norm(input, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        normed_feat = input / feat_norm
        label_view = label.view(-1, 1)
        w_norm = torch.norm(self.w, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        normed_w = self.w / w_norm
        cosine = torch.mm(normed_feat, normed_w).clamp(-1,1)
        #cosine = cosine.clamp(-1,1)
        #for i in range(input.shape[0]):
        #  cosine[i,int(label[i])] -= self.m
        #cosine_m = cosine
        ##cosine_m = cosine.scatter_add_(1, label_view, torch.ones_like(cosine, device=cosine.device) * (-self.m))
        tmp = self.gpu_tmp_map[cosine.device.index] 
        cosine_m = cosine.scatter_add_(1, label_view, tmp)
        #m = torch.zeros(cosine.size(), device=cosine.device).scatter_(1, label_view, self.m)
        #cosine_m = cosine - m
        cosine_m_s = self.s * cosine_m
        loss = self.ce(cosine_m_s, label)
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

        
