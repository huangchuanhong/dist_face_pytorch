import torch.nn as nn
from .backbones import DropBlockedSERegNet
from ..registry import BASE_MODEL
from ..utils import constant_init, normal_init,kaiming_init

@BASE_MODEL.register_module
class DropBlockedSERegnetModel(nn.Module):
    def __init__(self,
                 feature_dim,
                 last_channels=1008,
                 channels_per_group=16,
                 **kwargs):
        super(DropBlockedSERegnetModel, self).__init__()
        self.last_channels = last_channels
        self.backbone = DropBlockedSERegNet(**kwargs)
        self.gdc = nn.Conv2d(last_channels, last_channels, kernel_size=(7,7), groups=last_channels//channels_per_group, stride=(1,1), padding=(0,0), bias=False)
        self.bn = nn.BatchNorm2d(last_channels)
        #self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_channels, feature_dim)

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        kaiming_init(self.gdc)
        constant_init(self.bn, 1)
        normal_init(self.fc, mean=0, std=0.01)

    def forward(self, input):
        output = self.backbone(input)
        output = self.gdc(output)
        output = self.bn(output)
        output = output.view([-1, self.last_channels])
        output = self.fc(output)
        return output

    def train(self, mode):
        self.backbone.train(mode)
        self.bn.train(mode)
