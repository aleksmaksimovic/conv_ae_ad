#The code in this cell is taken from Lab07 with minor modifications
import numpy as np
import matplotlib.pyplot as plt
import requests
from pathlib import Path
import zipfile
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import os
import csv


class KolektorSDD2(Dataset):
    def __init__(self, path: Path, img_transform: transforms = None, train=True):
        self.images_path = path
        self.train=train
        
        self.img_paths, self.targets = self.read_data()
        self.img_transform = img_transform
        

        
            
    

    def read_data(self):
        img_paths, targets = [], []
        label_df=pd.read_csv(os.path.join(self.images_path, "image_labels.csv"), delimiter=";")
        
        
        for index, row in label_df.iterrows():
            if not(self.train==True and row['label']==1):
                img_paths.append(row['img_path'])
                targets.append(row['label'])

        return img_paths, targets

    

        

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        target = self.targets[idx]

        img = Image.open(img_path).convert("RGB")
        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, target

    def __len__(self):
        return len(self.img_paths)
    
    


