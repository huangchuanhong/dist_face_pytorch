import lmdb
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict

data_root_list = ['/mnt/data{}/huangchuanhong/datasets/weba1_splits_lmdb_{}'.format(i+1, i) for i in range(4)]
id_count = defaultdict(int)
for data_root in data_root_list:
    label_env = lmdb.open(os.path.join(data_root, 'train_label'), max_readers=8, readonly=True, lock=False,
                          readahead=False, meminit=False)
    label_txn = label_env.begin(write=False)
    for i in range(label_txn.stat()['entries']):
        if i & 5000 == 0:
            print('processed {} samples'.format(i))
        label = label_txn.get(str(i).encode())
        label = int(np.fromstring(label, dtype=np.int32))
        id_count[label] += 1
#print('num_samples={}'.format(len(labels)))
#with open('97w_label_count.txt', 'w') as f:
#    for label, count in id_count.items():
#        f.write('{} {}\n'.format(label, count))
figure = plt.figure()
label_counts = [count for i, count in id_count.items()]
max_ = max(label_counts)
min_ = min(label_counts)
print('max_count={}, min_count={}'.format(max_, min_))
plt.hist(label_counts, bins=max_ + 5)
plt.savefig('hist.jpg')
