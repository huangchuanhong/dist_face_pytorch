import math

import torch.nn as nn
import torch.utils.checkpoint as cp

from ...utils import constant_init, kaiming_init
from ....runner import load_checkpoint
from ...utils import build_norm_layer
from .layers import DropBlock2D

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 group_width=16,
                 downsample=None,
                 dropblock=False,
                 dropblock_prob=0.0,
                 block_size=5,
                 schedule_step=5000,
                 style='pytorch',
                 with_cp=False,
                 normalize=dict(type='BN')):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        self.dropblock = dropblock
        self.normalize = normalize
        self.group_width = group_width
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            normalize, planes * self.expansion, postfix=3)

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        if dropblock:
            self.dropblock1 = DropBlock2D(1 - dropblock_prob, block_size, schedule_step)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            groups=planes//self.group_width,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm2_name, norm2)
        if dropblock:
            self.dropblock2 = DropBlock2D(1 - dropblock_prob, block_size, schedule_step)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        if dropblock:
            self.dropblock3 = DropBlock2D(1 - dropblock_prob, block_size, schedule_step)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.normalize = normalize

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            if self.dropblock:
                out = self.dropblock1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            if self.dropblock:
                out = self.dropblock2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)
            if self.dropblock:
                out = self.dropblock3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            #out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   dropblock=False,
                   dropblock_prob=0.0,
                   block_size=5,
                   schedule_step=5000,
                   style='pytorch',
                   with_cp=False,
                   normalize=dict(type='BN')):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False),
            build_norm_layer(normalize, planes * block.expansion)[1],
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            dropblock=dropblock,
            dropblock_prob=dropblock_prob,
            block_size=block_size,
            schedule_step=schedule_step,
            style=style,
            with_cp=with_cp,
            normalize=normalize))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                stride=1,
                dilation=dilation,
                dropblock=dropblock,
                dropblock_prob=dropblock_prob,
                block_size=block_size,
                schedule_step=schedule_step,
                style=style,
                with_cp=with_cp,
                normalize=normalize))
    return nn.Sequential(*layers)


class DropBlockedRegNet(nn.Module):
    """RegNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        normalize (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """
    """
    GFlops groups not equals the groups of paper because of nnie
    """
    arch_settings = {
        '3.2GF': {'depth':[2, 6, 15, 2],
                  'width':[96, 192, 432, 1008]},
        '6.4GF': {'depth':[2, 4, 10, 1],
                  #'width':[168, 392, 784, 1624], origin},
                  'width':[144, 384, 768, 1632]}, 
    }

    def __init__(self,
                 gflops,
                 num_stages=4,
                 dropblock=False,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 dropblock_stages=(3, 4),
                 dropblock_prob=0.0,
                 block_size=5,
                 schedule_step=5000,
                 style='pytorch',
                 frozen_stages=-1,
                 normalize=dict(type='BN', frozen=False),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True):
        super(DropBlockedRegNet, self).__init__()
        if gflops not in self.arch_settings:
            raise KeyError('invalid gflops {} for regnet'.format(gflops))
        self.gflops = gflops
        self.block = Bottleneck
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.dropblock = dropblock
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.dropblock_stages = dropblock_stages
        self.dropblock_prob = dropblock_prob
        self.schedule_step = schedule_step
        self.style = style
        self.frozen_stages = frozen_stages
        self.normalize = normalize
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual

        self.stage_blocks = self.arch_settings[gflops]['depth'][:num_stages]
        self.stage_planes = self.arch_settings[gflops]['width'][:num_stages]
        self.inplanes = 64

        self._make_stem_layer()

        self.res_layers = []

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            if i in dropblock_stages:
                dropblock_i = True
            else:
                dropblock_i = False
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                self.stage_planes[i],
                num_blocks,
                stride=stride,
                dilation=dilation,
                dropblock=dropblock_i,
                dropblock_prob=self.dropblock_prob,
                block_size=block_size,
                schedule_step=schedule_step,
                style=self.style,
                with_cp=with_cp,
                normalize=normalize)
            self.inplanes = self.stage_planes[i]
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        # self._freeze_stages()

        # self.feat_dim = self.block.expansion * 64 * 2**(
        #     len(self.stage_blocks) - 1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self):
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.normalize, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)


    # def _freeze_stages(self):
    #     if self.frozen_stages >= 0:
    #         for m in [self.conv1, self.norm1]:
    #             for param in m.parameters():
    #                 param.requires_grad = False

    #     for i in range(1, self.frozen_stages + 1):
    #         m = getattr(self, 'layer{}'.format(i))
    #         for param in m.parameters():
    #             param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    # def train(self, mode=True):
    #     super(ResNet, self).train(mode)
    #     if mode and self.norm_eval:
    #         for m in self.modules():
    #             # trick: eval have effect on BatchNorm only
    #             if isinstance(m, nn.BatchNorm2d):
    #                 m.eval()
