import torch
from torch.utils.data import Dataset
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder

import lmdb
import numpy as np
import os
from PIL import Image

class EmoreLmdb(Dataset):
  def __init__(self, data_root):
    self.train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    data_env = lmdb.open(os.path.join(data_root, 'train_data'), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    label_env = lmdb.open(os.path.join(data_root, 'train_label'), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    self.data_txn = data_env.begin(write=False)
    self.label_txn = label_env.begin(write=False)

  def __getitem__(self, index):
    img_bytes = self.data_txn.get(str(index+1).encode())
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((112, 112, 3))
    img = Image.fromarray(img)
    img = self.train_transform(img)
    label = self.label_txn.get(str(index+1).encode())
    label = int(np.frombuffer(label, dtype=np.int64))
    return {'img': img, 'label': label}

  def __len__(self):
    return 5822653
   
  
if __name__ == '__main__':
  emore = EmoreLmdb('/mnt/data4/huangchuanhong/datasets/faces_emore/lmdb') 
  data = emore[0]
  img = data['img']
  label = data['label']
  print('type(img) = {}'.format(type(img)))
  print('img.dtype={}'.format(img.dtype))
  print('img.shape = {}'.format(img.shape))
  print('label={}'.format(label))
  
