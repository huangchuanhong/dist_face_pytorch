import math
import torch
import torch.nn as nn
from ..registry import TOP_MODEL
from ..utils import normal_init
from .top_model import TopModel
import threading

@TOP_MODEL.register_module
class CircleLossMP(TopModel):
    def __init__(self, num_classes, m=0.25, gamma=256, feature_dim=192):
        super(CircleLossMP, self).__init__(feature_dim,num_classes)
        self.margin = m
        self.gamma = gamma
        self.device_count = torch.cuda.device_count()
        self.ws = nn.ParameterList([])
        self.num_classes = num_classes
        for i in range(self.device_count):
            self.ws.append(torch.nn.Parameter(
                torch.randn(feature_dim, self.classes_nums[i], requires_grad=True, device='cuda:{}'.format(i))))
        self.init_weights()
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
        y_true_i = torch.zeros((cosine.size()[0], cosine.size()[1] + 1), device=device_i).scatter_(1, label_i_view, 1)[:, :-1]
        y_pred_part = (y_true_i * (alpha_p * (cosine - self.Delta_p)) +
                       (1 - y_true_i) * (alpha_n * (cosine - self.Delta_n))) * self.gamma
        return y_pred_part

