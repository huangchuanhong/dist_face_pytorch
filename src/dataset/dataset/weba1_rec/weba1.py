import torch
import numpy as np

class RecDataLoder(object):
    def __init__(self, dataloader, dataloader_lens):
        self.dataloader = dataloader
        self.dataloader_lens = dataloader_lens
        self._iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        data_batch = self.dataloader.next()
        data_batch_data = torch.from_numpy(data_batch.data[0].asnumpy().reshape((-1, 3, 108, 108)))
        data_batch_label = torch.from_numpy(data_batch.label[0].asnumpy().astype(np.int64))
        data_batch = {'img': data_batch_data, 'label': data_batch_label}
        self._iter += 1
        if self._iter == self.dataloader_lens:
            self.dataloader.reset()
            self._iter = 0
        return data_batch

    def __len__(self):
        return self.dataloader_lens
    

