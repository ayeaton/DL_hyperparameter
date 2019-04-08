import random
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import torchvision.transforms as transforms


standard_transform = transforms.Compose([
    transforms.Scale(299),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])])

class dataset(Dataset):
    """
    Data loader for NYU FFPE data. Reads in

    file_path: path to file containing the list of tuples
    split: Train, Val, or Test
    tranformer: transform
    classes: list of classes to use, or "all"
    """
    def __init__(self, file_path, split, transformer, classes):

        self.file_path = file_path
        self.split = split
        self.transformer = transformer

        with open(self.file_path, 'r') as filehandle:
            metadata_list = json.load(filehandle)

        metadata_df = pd.DataFrame(metadata_list, columns =['Path', 'label', 'slide', 'label_num', 'split'])
        metadata_split = metadata_df[metadata_df['split'] == self.split]

        if classes != "all":
            metadata_split = metadata_split[metadata_split['label'].isin(classes)]

        self.image_paths = metadata_split['Path']
        self.labels = metadata_split['label_num']

    def __len__(self):
        # return size of dataset
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(self.image_paths.iloc[idx])  # PIL image
        label = self.labels.iloc[idx]
        image = self.transformer(image)

        return image, label
