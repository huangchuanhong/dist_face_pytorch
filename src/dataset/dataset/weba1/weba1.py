import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
import random
from ctypes import *
from torch.utils.data import Dataset

import logging
import time
_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=_format, level='DEBUG')
_logger = logging.getLogger('debug')


class WebA1(Dataset):

    def __init__(self,
                 data_root,
                 list_file,
                 transform_config,
                 test_mode=False):
        start = time.time() 
        self.train_transform = trans.Compose([
                trans.Resize(108),
                trans.RandomHorizontalFlip(),
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        self.samples = []
        with open(list_file) as f:
            for line in f.readlines():
                splits = line.strip().split(' ')
                self.samples.append((os.path.join(data_root, splits[0]), int(splits[1])))
        _logger.info('dataset  pytorch transform  init, time used:{}'.format(time.time() - start))
        ##self.ds = ImageFolder(data_root + 'train_10000', train_transform) 
        
 
    def __getitem__(self, index):
        path, target = self.samples[index]
        trans_img = self._transform(path)
        #print('trans_img.device={}'.format(trans_img.device))
        #trans_img = torch.ones([3, 108, 108])
        #data = self.ds[index]
        #return {'img':data[0], 'label': data[1]}
        #return {'img': trans_img, 'label': 1}
        return {'img':trans_img, 'label':target}

    def _transform(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            trans_img = self.train_transform(img)
            return trans_img

    def __len__(self):
        #return len(self.ds)
        return len(self.samples)
        
if __name__ == '__main__':
    data_root = '/mnt/data3/huangchuanhong/253_data/dataset/face/data/recover_97w' 

    dataset = WebA1(data_root, 'train_xxs.lst', transform_config=None)
    print('len(dataset)=', len(dataset))
    for i in range(len(dataset)):
        trans_img = dataset[i]['img']
        print('trans_img.type={}'.format(type(trans_img)))
        cv2.imwrite('debug/{}.jpg'.format(i), trans_img.numpy().transpose((1,2,0)) * 255)
