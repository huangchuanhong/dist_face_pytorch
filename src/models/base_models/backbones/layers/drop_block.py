import torch
import torch.nn.functional as F
from torch import nn


class DropBlock2D(nn.Module):
    r"""Randomly zeroes spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        keep_prob (float, optional): probability of an element to be kept.
        Authors recommend to linearly decrease this value from 1 to desired
        value.
        block_size (int, optional): size of the block. Block size in paper
        usually equals last feature map dimensions.
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, keep_prob=0.9, block_size=7, schedule_steps=None):
        super(DropBlock2D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size
        if schedule_steps:
            self.curr_keep_prob = 1.
            self.stepsize_ = (1 - keep_prob) / schedule_steps
            self.step_ = 0
            self.schedule_steps = schedule_steps
        else:
            self.curr_keep_prob = keep_prob
        self.training = False

    def forward(self, input):
        if not self.training or self.keep_prob == 1:
            return input
        gamma = (1. - self.curr_keep_prob) / self.block_size ** 2
        for sh in input.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        mask = torch.bernoulli(torch.ones_like(input) * gamma)
        mask = F.max_pool2d(
            input=mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )
        mask = (mask < 1).to(device=input.device, dtype=input.dtype)
        if self.step_ < self.schedule_steps:
            self.curr_keep_prob -= self.stepsize_
        else:
            self.step_ += 1
        return input * mask * mask.numel() /mask.sum() #TODO input * mask * self.keep_prob ?

    def train(self, mode=True):
        super(DropBlock2D, self).train(mode=mode)
        self.training = True
