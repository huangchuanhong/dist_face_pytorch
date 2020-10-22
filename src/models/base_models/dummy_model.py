import torch.nn as nn
from ..registry import BASE_MODEL

@BASE_MODEL.register_module
class DummyModel(nn.Module):
    def __init__(self, feature_dim, **kwargs):
        super(DummyModel, self).__init__()
        pass

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)

    def forward(self, input):
        output = self.backbone(input)
        return output

    def train(self, mode):
        self.backbone.train(mode)
