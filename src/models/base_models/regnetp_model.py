import torch.nn as nn
from .backbones import RegNetP
from ..registry import BASE_MODEL
from ..utils import constant_init, normal_init,kaiming_init
from .regnet_model import RegnetModel

@BASE_MODEL.register_module
class RegnetPModel(nn.Module):
    def __init__(self,
                 feature_dim,
                 **kwargs):
        super(RegnetPModel, self).__init__()
        self.backbone = RegNetP(**kwargs)
        self.gdc = nn.Conv2d(1008, 1008, kernel_size=(7,7), groups=1008//16, stride=(1,1), padding=(0,0), bias=False)
        self.bn = nn.BatchNorm2d(1008)
        #self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1008, feature_dim)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        kaiming_init(self.gdc)
        constant_init(self.bn, 1)
        normal_init(self.fc, mean=0, std=0.01)

    def forward(self, input):
        output = self.backbone(input)
        output = self.gdc(output)
        output = self.bn(output)
        output = output.view([-1, 1008])
        output = self.fc(output)
        return output

    def train(self, mode):
        self.backbone.train(mode)
        self.bn.train(mode)

