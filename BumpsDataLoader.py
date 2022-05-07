# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class BumpDataset(Dataset):
    def __init__(self,root = './data/', npoints=2048, split='train'):
        self.npoints = npoints
        self.root = root
        
        self.cat = {'Road': 'roadern'}
        self.classes_original = {'Road': 0}
        
        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])


        # Find all filenames corresponding to the current category
        dir_point = os.path.join(self.root, self.cat[0])
        fns = sorted(os.listdir(dir_point))

        if split == 'trainval':
            fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
        elif split == 'train':
            fns = [fn for fn in fns if fn[0:-4] in train_ids]
        elif split == 'val':
            fns = [fn for fn in fns if fn[0:-4] in val_ids]
        elif split == 'test':
            fns = [fn for fn in fns if fn[0:-4] in test_ids]
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)
        
        self.fns = [os.path.join(dir_point, fn) for fn in self.fns]

        self.seg_classes = {'Road': [0, 1, 2]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            point_set, seg = self.cache[index]
        else:
            fn = self.fns[index]
            data = np.loadtxt(fn).astype(np.float32)
            point_set = data[:, 0:3] 
            seg = data[:, -1].astype(np.int32)
            
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, seg

    def __len__(self):
        return len(self.fns)

# Test implementation
if __name__ == '__main__':
    import torch


    data_root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    npoints = 1024
    data = BumpDataset(root=data_root, npoints=1024, split='trainval')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for points, label, target in DataLoader:
        print('points:', points.shape)
        print('label :', label.shape)
        print('target:', target.shape)
        break

