import torch.nn as nn
from .backbones import ResNest
from ..registry import BASE_MODEL
from ..utils import constant_init, normal_init, kaiming_init

@BASE_MODEL.register_module
class ResNestModel(nn.Module):
    def __init__(self,
                 feature_dim,
                 **kwargs):
        super(ResNestModel, self).__init__()
        self.backbone = ResNest(**kwargs)
        self.gdc = nn.Conv2d(2048, 2048, groups=2048//16, kernel_size=(7, 7), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, feature_dim)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        kaiming_init(self.gdc)
        constant_init(self.bn, 1)
        #normal_init(self.fc, std=0.01)

    def forward(self, input):
        output = self.backbone(input)
        output = self.gdc(output)
        output = self.bn(output)
        output = output.view([-1, 2048])
        output = self.fc(output)
        return output

    def train(self, mode):
        self.backbone.train(mode)
        self.bn.train(mode)
