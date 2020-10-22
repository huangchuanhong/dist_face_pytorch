import torch
from torch.utils.data import Dataset
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder

import lmdb
import numpy as np
import cv2
import os
from PIL import Image, ImageEnhance
import random

class WebA1Lmdb(Dataset):
    def __init__(self, data_root, data_aug=False):
        self.train_transform = trans.Compose([
                trans.CenterCrop(112),
                trans.RandomCrop(108),
                trans.RandomHorizontalFlip(),
                trans.ToTensor(),
                trans.Normalize([0.482352, 0.45490, 0.40392], [0.392157, 0.392157, 0.392157])
            ])
        data_env = lmdb.open(os.path.join(data_root, 'train_data'), max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
        label_env = lmdb.open(os.path.join(data_root, 'train_label'), max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
        self.data_txn = data_env.begin(write=False)
        self.label_txn = label_env.begin(write=False)
        if data_aug:
            extra_label_env = lmdb.open(os.path.join(data_root, 'train_extra_label'), max_readers=8, readonly=True, lock=False, readahead=False, meminit=False)
            self.extra_label_txn = extra_label_env.begin(write=False)
            self.brightness_level_enhance_range = {
                '0': [1, 2.25],
                '1': [1, 2.25],
                '2': [1, 2],
                '3': [0.7, 1.75],
                '4': [0.6, 1.5], 
                '5': [0.5, 1.3],
                '6': [0.4, 1.2], 
                '7': [0.35, 1.1],
                '8': [0.3, 1.65],
                '9': [0.3, 1],
            }
        self.data_aug = data_aug
  
    def __getitem__(self, index):
        img_bytes = self.data_txn.get(str(index).encode())
        img = np.fromstring(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        #img = img[:, :, ::-1]
        #img = Image.fromarray(img)
        if self.data_aug:
            img = self.random_augmentation(img, self.extra_label_txn.get(str(index).encode()))
        else:
            img = img[:, :, ::-1]
            img = Image.fromarray(img)
        img = self.train_transform(img)
        label = self.label_txn.get(str(index).encode())
        label = int(np.fromstring(label, dtype=np.int32))
        return {'img': img, 'label': label}

    def random_augmentation(self, img, extra_label):
        blur_level = int(extra_label[1])
        if blur_level and random.random() < 0.8:
            size = int(random.random() * 15)
            kernel = np.zeros((size, size))
            kernel[int((size-1) / 2), :] = 1./ size
            img = cv2.filter2D(img, -1, kernel)
        img = img[:, :, ::-1]
        img = Image.fromarray(img)
        brightness_level = int(extra_label[0])
        enhance_range = self.brightness_level_enhance_range[str(brightness_level)]
        scale = random.random() * (enhance_range[1] - enhance_range[0]) + enhance_range[0]
        img = ImageEnhance.Brightness(img).enhance(scale)
        return img

 
    #def random_augmentation(self, img):
    #    prob = random.random()
    #    if prob < self.constrast_prob:
    #        img = ImageEnhance.Contrast(img).enhance(1 + self.contrast_thr * random.choice([-1, 1]))
    #    prob = random.random()
    #    if prob < self.sharpness_prob:
    #        img = ImageEnhance.Sharpness(img).enhance(1 + self.sharpness_thr * random.choice([-1, 1]))
    #    prob = random.random()
    #    if prob < self.brightness_prob:
    #        img = ImageEnhance.Brightness(img).enhance(1 + self.brightness_thr * random.choice([-1, 1]))
    #    prob = random.random()
    #    if prob < self.color_prob:
    #        img = ImageEnhance.Color(img).enhance(1 + self.brightness_thr * random.choice([-1, 1]))
    #    return img

    def __len__(self):
        return self.label_txn.stat()['entries']
  

class WebA1ConcateLmdb(Dataset):
    def __init__(self, data_root_list, data_aug=False):
        datasets = []
        self._len = 0
        for data_root in data_root_list:
            lmdb_dataset = WebA1Lmdb(data_root, data_aug)
            datasets.append(lmdb_dataset)
            self._len += len(lmdb_dataset)
        self.concate_dataset = torch.utils.data.ConcatDataset(datasets)
    
    def __getitem__(self, index):
        return self.concate_dataset[index]
   
    def __len__(self):
        return self._len 
   
  
if __name__ == '__main__':
    #emore = EmoreLmdb('/mnt/data4/huangchuanhong/datasets/faces_emore/lmdb') 
    #data = emore[0]
    #img = data['img']
    #label = data['label']
    #print('type(img) = {}'.format(type(img)))
    #print('img.dtype={}'.format(img.dtype))
    #print('img.shape = {}'.format(img.shape))
    #print('label={}'.format(label))
  
    data_root_list = []
    for i in range(4):
        data_root_list.append('/mnt/data{}/huangchuanhong/datasets/weba1_splits_lmdb_{}'.format(i+1, i))
    concat_lmdb = WebA1ConcateLmdb(data_root_list, data_aug=True)
    for i in range(1000):
        concat_lmdb[i]
    print(len(concat_lmdb))
    #print(concat_lmdb[0])
    
