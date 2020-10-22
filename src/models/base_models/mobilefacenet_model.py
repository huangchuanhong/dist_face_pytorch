import torch.nn as nn
from .backbones import MobileFaceNet
from ..registry import BASE_MODEL

@BASE_MODEL.register_module
class MobileFaceNetModel(nn.Module):
    def __init__(self, feature_dim, **kwargs):
        super(MobileFaceNetModel, self).__init__()
        self.backbone = MobileFaceNet(feature_dim)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)

    def forward(self, input):
        output = self.backbone(input)
        return output

    def train(self, mode):
        self.backbone.train(mode)
