import torch.distributed as dist
import torch

def onebase_1top_dataloader(dataset, batch_size, num_workers, mode='train'):
    if mode == 'train':
        drop_last = True
        shuffle = True
    elif mode == 'val':
        drop_last = False
        shuffle = False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader