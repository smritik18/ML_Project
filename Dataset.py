import torch
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_file)
        self.transform = transform
        self.label_mapping = {label: index for index, label in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")}

        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.data_df.iloc[idx, 0])
        image = Image.open(img_name)
        label_str = self.data_df.iloc[idx, 1]
        label = self.label_mapping[label_str]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label)