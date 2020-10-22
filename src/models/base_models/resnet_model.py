import torch.nn as nn
from .backbones import ResNet
from ..registry import BASE_MODEL
from ..utils import constant_init, normal_init

@BASE_MODEL.register_module
class ResnetModel(nn.Module):
    def __init__(self,
                 feature_dim,
                 **kwargs):
        super(ResnetModel, self).__init__()
        self.backbone = ResNet(**kwargs)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, feature_dim)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        constant_init(self.bn, 1)
        normal_init(self.fc, std=0.01)

    def forward(self, input):
        output = self.backbone(input)
        output = self.pool(output)#.squeeze()
        output = self.bn(output).squeeze()
        output = self.fc(output)
        return output

    def train(self, mode):
        self.backbone.train(mode)
        #self.bn.eval()
