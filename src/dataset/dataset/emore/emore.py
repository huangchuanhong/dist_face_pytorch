import torch
from torch.utils.data import Dataset
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder

class Emore(Dataset):
  def __init__(self, data_root, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], random_crop=False):
    if random_crop:
        train_transform = trans.Compose([
            trans.RandomCrop(108),
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            trans.Normalize(mean, std)
        ])
    else:
        train_transform = trans.Compose([
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            trans.Normalize(mean, std)
        ])

    print('before ImageFolder')
    self.ds = ImageFolder(data_root, train_transform)
    self.labels = self.ds.targets
    print('num_samples={}'.format(len(self.labels)))
    print('after ImageFolder')

  def __getitem__(self, index):
    data = self.ds[index]
    return {'img': data[0], 'label': data[1]}

  def __len__(self):
    return len(self.ds)
   
  
if __name__ == '__main__':
  emore = Emore('/mnt/data4/huangchuanhong/datasets/faces_emore/imgs_tmp') 
  dataloader = torch.utils.data.DataLoader(emore, batch_size=2, num_workers=1)
  dataloader = iter(dataloader)
  print(next(dataloader))
