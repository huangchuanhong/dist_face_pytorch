import math
import torch
import torch.nn as nn
from ..registry import TOP_MODEL
from ..utils import normal_init
from .top_model import TopModel
import threading

@TOP_MODEL.register_module
class AmSoftmax1DeviceMTLogSumExpCEModel(TopModel):
    def __init__(self, feature_dim, num_classes, m=0.35, s=30):
        super(AmSoftmax1DeviceMTLogSumExpCEModel, self).__init__(feature_dim, num_classes)
        self.m = m
        self.s = s
        self.device_count = torch.cuda.device_count()
        self.ws = nn.ParameterList([])
        self.num_classes = num_classes
        # self.probs = torch.zeros([batch_size, num_classes], device='cuda:0')
        for i in range(self.device_count):
            self.ws.append(torch.nn.Parameter(
                torch.randn(feature_dim, self.classes_nums[i], requires_grad=True, device='cuda:{}'.format(i))))
            # self.ces.append(nn.CrossEntropyLoss().to('cuda:{}'.format(i)))
        self.init_weights()

    def init_weights(self, pretrained=None):
        for w in self.ws:
            w.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

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

