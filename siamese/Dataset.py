#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   Dataset.py
@Time    :   2022/04/25 22:35:18
@Author  :   xxxx
@Version :   1.0
@Contact :   xxxx
@License :   xxxx
@Desc    :   <awaiting description>
'''

import pandas as pd
from PIL import Image
from pathlib import Path
import os
import torch

class Dataset:
    def __init__(self, data_dir=None, mode=None, transform=None): 
        self.data_dir = Path(data_dir)
        self.data_df = pd.read_csv(os.path.join(data_dir, f"{mode}.csv"), \
            names=["imageA", "imageB", "label"])
        self.transform = transform
        self.mode = mode
        
    def __getitem__(self, index):
        imageA_path = str(self.data_dir / f"{self.data_df.at[index, 'imageA']}")
        imageB_path = str(self.data_dir / f"{self.data_df.at[index, 'imageB']}")
    
        imga = Image.open(imageA_path).convert("L")
        imgb = Image.open(imageB_path).convert("L")
        
        if self.transform:
            imga = self.transform(imga)
            imgb = self.transform(imgb)
        return imga, imgb, torch.tensor([self.data_df.at[index, 'label']], dtype=torch.float32)
        
    def __len__(self):
        return len(self.data_df)
    

if __name__ == '__main__':
    data_dir = "/media/lx/data/code/siamese/dataset"
    data = Dataset(data_dir, "train")
    t = torch.tensor([1], dtype=torch.float32)
    print(t)