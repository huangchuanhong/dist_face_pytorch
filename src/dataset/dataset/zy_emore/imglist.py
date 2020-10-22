import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from PIL import Image
import torchvision.transforms as trans
class ImgList(Dataset):
    def __init__(self,
                 data_root,
                 list_file,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 random_crop=False, 
                 test_mode=False):
        self.samples = []
        self.labels = []
        if random_crop:
            self.transform = trans.Compose([
                trans.RandomCrop(108),
                trans.RandomHorizontalFlip(),
                trans.ToTensor(),
                trans.Normalize(mean, std)
            ])
        else:
            self.transform = trans.Compose([
                trans.RandomHorizontalFlip(),
                trans.ToTensor(),
                trans.Normalize(mean, var)
            ])
        with open(list_file) as f:
            for line in f.readlines():
                splits = line.strip().split()
                self.samples.append((os.path.join(data_root, splits[0]), int(splits[1])))
                self.labels.append(int(splits[1]))

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = cv2.imread(path)
        img = Image.fromarray(img)
        trans_img = self.transform(img)
        return {'img':trans_img, 'label':target}

    def __len__(self):
        return len(self.samples)

