import random
from random import choices
import numpy as np
import pandas as pd
from collections import Counter

import torch
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split

label_to_class = {'MildDemented': 0, 'ModerateDemented': 1, 'NonDemented': 2, 'VeryMildDemented': 3}  # Adjust this as needed


class AlzheimerDataset(Dataset):
    def __init__(self, base_dir, phase, label_to_class, transform=None):
        self.data = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        dir_name = os.path.join(base_dir, phase)
        for label_name in os.listdir(dir_name):
            label_path = os.path.join(dir_name, label_name)
            class_index = label_to_class[label_name]
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                self.data.append((img_path, class_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_index = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, class_index

def setup_data(args):

    # Initialize your datasets
    train_dataset = AlzheimerDataset(args.base_dir_img , 'train', label_to_class)
    test_dataset = AlzheimerDataset(args.base_dir_img , 'test', label_to_class)
    if args.mix_img == True:
        # Combine the datasets
        combined_dataset = ConcatDataset([train_dataset, test_dataset])

        # Calculate the lengths for split: 
        total_size = len(combined_dataset)
        print(total_size)
        train_size = int(args.split_data * total_size)
        valid_size = total_size - train_size

        # Split the dataset
        train_dataset, valid_dataset = random_split(combined_dataset, [train_size, valid_size])

    # Setup the DataLoader for both the training and validation datasets
    train_dataset = DataLoader(train_dataset, batch_size= args.batch_size , shuffle=True, num_workers=8)
    test_dataset = DataLoader(valid_dataset, batch_size= args.batch_size , shuffle=False, num_workers=8)
    return train_dataset , test_dataset