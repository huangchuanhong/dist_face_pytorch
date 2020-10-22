#TODO: to be completed

from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import torch
import math
from collections import defaultdict
import random
import numpy as np

class BalancedBatchSampler(Sampler):
    """
    samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples

    example:
    -------------------------
    from torch.utils.data import Dataset
    class FooDataset(Dataset):
        def __init__(self):
            self.data = list(range(10))
            self.label = [1,1,1,1,2,2,3,3,3,4]
        def __getitem__(self, idx):
            return {'data': self.data[idx], 'label': self.label[idx]}
        def __len__(self):
            return 10

    dataset = FooDataset()
    sampler = BalancedBatchSampler(dataset, 2, 2)
    for x in sampler:
        print(x)
    -------------------------
    """

    def __init__(self, dataset, n_classes, n_samples):
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.labels = []
        # for i, sample in enumerate(self.dataset):
        #     self.labels.append(sample['label'])
        self.labels = self.dataset.labels

        self.ori_labels_set = list(set(self.labels))
        self.label_to_indices = {label:[] for label in self.ori_labels_set}
        for nn, label in enumerate(self.labels):
            self.label_to_indices[label].append(nn)
        #self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
        #                         for label in self.labels_set}
        self.labels_set = []
        for k,v in self.label_to_indices.items():
            if len(v) >= n_samples:
                self.labels_set.append(k)
        random.shuffle(self.labels_set)
        print("dataset has {} classes".format(len(self.labels_set)))
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.used_label_count = 0
        self.count = 0
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes
        print("BalanceBatchSampler init")
    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            classes = self.labels_set[self.used_label_count:self.used_label_count+self.n_classes]
            self.used_label_count += self.n_classes
            if self.used_label_count + self.n_classes > len(self.labels_set):
                  np.random.shuffle(self.labels_set)
                  self.used_label_count = 0
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            random.shuffle(indices)
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class DistributedBalancedBatchSampler(Sampler):
    """
    samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples

    example:
    ---------------------
    import os
    from torch.utils.data import Dataset
    import socket
    def get_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        return ip
    ip = get_ip().split('.')[-1]
    print('ip={}'.format(ip))
    ip_rank_map = {
        '35': 0,
        '245': 1
    }

    rank = ip_rank_map[ip]
    size = len(ip_rank_map)

    def init_process(rank, size):
        backend = 'nccl'
        rank = rank
        size = size
        os.environ['MASTER_ADDR'] = '10.58.122.35'
        os.environ['MASTER_PORT'] = '123456'
        os.environ['WORLD_SIZE'] = str(size)
        os.environ['RNAK'] = str(rank)
        dist.init_process_group(backend, rank=rank, world_size=size)
    class FooDataset(Dataset):
        def __init__(self):
            self.data = list(range(10))
            self.label = [1,1,1,1,2,2,3,3,3,4]
        def __getitem__(self, idx):
            return {'data': self.data[idx], 'label': self.label[idx]}
        def __len__(self):
            return 10

    init_process(rank, size)
    print('after init_process')
    dataset = FooDataset()
    sampler = DistributedBalancedBatchSampler(dataset, 2, 1, size, rank)
    for x in sampler:
        print(x)
    import time
    time.sleep(100000)
    -------------------
    """

    def __init__(self, dataset, n_classes, n_samples, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.num_replicas = num_replicas
        self.rank = rank

        self.total_labels = self.dataset.labels
        # for i, sample in enumerate(self.dataset):
        #     self.total_labels.append(sample['label'])
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.total_labels):
            self.label_to_indices[label].append(idx)
        total_labels_set = []
        for label_, indices_ in self.label_to_indices.items():
            if len(indices_) >= n_samples:
                total_labels_set.append(label_)
        total_labels_set.sort()
        part_labels_num = math.ceil(len(total_labels_set) / self.num_replicas * 1.0)
        self.label_sets_list = [total_labels_set[part_labels_num * i: part_labels_num * (i + 1)] for i in range(self.num_replicas)]
        if len(self.label_sets_list[-1]) < part_labels_num:
            self.label_sets_list[-1].extend(self.label_sets_list[-1][:part_labels_num - len(self.label_sets_list[-1])])
        self.labels_list = []
        for rank in range(self.num_replicas):
            labels_ = []
            for label_ in self.label_sets_list[rank]:
                labels_.extend(self.label_to_indices[label_])
            self.labels_list.append(labels_)
        self.n_dataset = max([len(x) for x in self.labels_list])
        self.labels_set = self.label_sets_list[self.rank]
        self.labels = self.labels_list[self.rank]
        random.shuffle(self.labels_set)
        for l in self.labels_set:
            random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.used_label_count = 0
        self.count = 0
        self.batch_size = self.n_samples * self.n_classes
        self.epoch = 0

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            classes = self.labels_set[self.used_label_count:self.used_label_count+self.n_classes]
            self.used_label_count += self.n_classes
            if self.used_label_count + self.n_classes > len(self.labels_set):
                  np.random.shuffle(self.labels_set)
                  self.used_label_count = 0
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            #print("-----------bs--------\n",len(indices),indices)
            random.shuffle(indices)
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

if __name__ == '__main__':
    # import os
    # from torch.utils.data import Dataset
    # import socket
    # def get_ip():
    #     try:
    #         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #         s.connect(('8.8.8.8', 80))
    #         ip = s.getsockname()[0]
    #     finally:
    #         s.close()
    #     return ip
    # ip = get_ip().split('.')[-1]
    # print('ip={}'.format(ip))
    # ip_rank_map = {
    #     '35': 0,
    #     '245': 1
    # }
    #
    # rank = ip_rank_map[ip]
    # size = len(ip_rank_map)
    #
    # def init_process(rank, size):
    #     backend = 'nccl'
    #     rank = rank
    #     size = size
    #     os.environ['MASTER_ADDR'] = '10.58.122.35'
    #     os.environ['MASTER_PORT'] = '123456'
    #     os.environ['WORLD_SIZE'] = str(size)
    #     os.environ['RNAK'] = str(rank)
    #     dist.init_process_group(backend, rank=rank, world_size=size)
    # class FooDataset(Dataset):
    #     def __init__(self):
    #         self.data = list(range(10))
    #         self.label = [1,1,1,1,2,2,3,3,3,4]
    #     def __getitem__(self, idx):
    #         return {'data': self.data[idx], 'label': self.label[idx]}
    #     def __len__(self):
    #         return 10
    #
    # init_process(rank, size)
    # print('after init_process')
    # dataset = FooDataset()
    # sampler = DistributedBalancedBatchSampler(dataset, 2, 2, size, rank)
    # for x in sampler:
    #     print(x)
    # print('len(sampler)={}'.format(len(sampler)))
    # import time
    # time.sleep(100000)

    from torch.utils.data import Dataset
    class FooDataset(Dataset):
        def __init__(self):
            self.data = list(range(10))
            self.label = [1,1,1,1,2,2,3,3,3,4]
        def __getitem__(self, idx):
            return {'data': self.data[idx], 'label': self.label[idx]}
        def __len__(self):
            return 10

    dataset = FooDataset()
    sampler = BalancedBatchSampler(dataset, 2, 2)
    for x in sampler:
        print(x)