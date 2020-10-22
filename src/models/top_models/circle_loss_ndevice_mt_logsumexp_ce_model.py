import math
import torch
import torch.nn as nn
from ..registry import TOP_MODEL
from ..utils import normal_init
from .ndevice_top_model import NdeviceTopModel
import threading

@TOP_MODEL.register_module
class CircleLossNTopMP(NdeviceTopModel):
    def __init__(self, batch_size, feature_dim, num_classes, top_model_ranks, top_model_gpus, top_group,
                 m=0.25, gamma=256., ** kwargs):
        super(CircleLossNTopMP, self).__init__(batch_size, feature_dim, num_classes, top_model_ranks, top_model_gpus, top_group,
                                               ** kwargs)
        self.margin = m
        self.gamma = gamma
        self.num_classes = num_classes
        self.O_p = 1 + self.margin
        self.O_n = -self.margin
        self.Delta_p = 1 - self.margin
        self.Delta_n = self.margin

    def init_weights(self, pretrained=None):
        for w in self.ws:
            w.data.normal_(0, 0.001)

    def get_probs_i(self, cosine, label_i, device_i):
        alpha_p = (self.O_p - cosine.detach()).clamp(min=0)
        alpha_n = (cosine.detach() - self.O_n).clamp(min=0)
        label_i_view = label_i.view(-1, 1)
        y_true_i = torch.zeros((cosine.size()[0], cosine.size()[1] + 1), device=device_i).scatter_(1, label_i_view, 1)[
                   :, :-1]
        y_pred_part = (y_true_i * (alpha_p * (cosine - self.Delta_p)) +
                       (1 - y_true_i) * (alpha_n * (cosine - self.Delta_n))) * self.gamma
        return y_pred_part

