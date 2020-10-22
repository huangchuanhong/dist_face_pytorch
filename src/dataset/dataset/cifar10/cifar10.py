import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10


class Cifar10(CIFAR10):
    def __init__(self, **kwargs):
        self.cifar10 = CIFAR10(**kwargs)
   
    def __getitem__(self, idx):
        trans_img, target = self.cifar10[idx]
        return {'img':trans_img, 'label':target}

    def __len__(self):
        return len(self.cifar10)



