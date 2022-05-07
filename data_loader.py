import os
import numpy as np
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    
    def __init__(self, annotations_file, pc_dir, npoints=1024, split="train", classes="reg_dump", mode="TRAIN", random_select=True,
                 logger=None, rcnn_training_roi_dir=None, rcnn_training_feature_dir=None, rcnn_eval_roi_dir=None,
                 rcnn_eval_feature_dir=None, gt_database_dir=None, transform=None, target_transform=None):
        # TODO: Create method to read all annotations (comes in different json files)
        self.labels = annotations_file
        self.pc_dir = pc_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        pc_path = os.path.join(self.pc_dir, self.labels.iloc[idx, 0])
        # TODO: Create method to read point cloud
        pc = pc_path
        label = self.labels.iloc[idx, 1]
        if self.transform:
            pc = self.transform(pc)
        if self.target_transform:
            label = self.target_transform(label)
        return pc, label
    
    
