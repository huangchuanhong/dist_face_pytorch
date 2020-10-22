import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

class WebA1Pytorch(Dataset):
    def __init__(self, 
                 data_root,
                 list_file,
                 transform_config,
                 test_mode=False):
        self.samples = []
        with open(list_file) as f:
            for line in f.readlines():
                splits = line.strip().split(' ')
                self.samples.append((os.path.join(data_root, splits[0]), int(splits[1])))

    def __getitem__(self, index):
        path, target = self.samples[index]
        trans_img = self._transform(path)
        return {'img':trans_img, 'label':target}

    def _transform(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (108, 108))
        img = (img - 128) / 255
        img = img.astype(np.float32)
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        return img
 
    def __len__(self):
        return len(self.samples)

