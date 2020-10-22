import math
import torch
import torch.nn as nn
from ..registry import TOP_MODEL
from ..utils import normal_init
from .ndevice_top_model import NdeviceTopModel
import threading

@TOP_MODEL.register_module
class AmSoftmaxNDeviceMTLogSumExpCEModel(NdeviceTopModel):
    def __init__(self, batch_size, feature_dim, num_classes, top_model_ranks, top_model_gpus, top_group, m=0.35, s=30):
        super(AmSoftmaxNDeviceMTLogSumExpCEModel, self).__init__(
            batch_size, feature_dim, num_classes, top_model_ranks, top_model_gpus, top_group
        )
        self.m = m
        self.s = s

    def init_weights(self, pretrained=None):
        for w in self.ws:
            w.data.normal_(0, 0.001)

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

