import torch.nn as nn
from .backbones import resnext50
from ..registry import BASE_MODEL

@BASE_MODEL.register_module
class ResNext50Model(nn.Module):
    def __init__(self, feature_dim, **kwargs):
        super(ResNext50Model, self).__init__()
        self.backbone = resnext50(embedding_size=feature_dim, num_group=32)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)

    def forward(self, input):
        output = self.backbone(input)
        return output

    def train(self, mode):
        self.backbone.train(mode)
