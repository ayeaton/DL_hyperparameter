import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import glob
from numpy import array
import numpy as np
import torch

class images_dataset(Dataset):
    def __init__(self, data_dir, image_list_file, time_tile):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            time_tile: number of tiles along time dimension
            vid_range_start: index of the first image to use. Indexed from the end
            vid_range_end: index of the last image to use. Indexed from the end

        Further explanation:
            The length of the video loaded in will be vid_range_start-vid_range_end. This video is
            further split by the time_tile. The length of the video must be evenly divisible by the time_tile. For
            example -- vid_range_start=10, and vid_range_end=2 and time_tile=2. The dataloader will grab the last 10 images
            of the video up to the last 2, for a total of 8 images. Then, these images will be split into two tiles
            resulting in 2, 4 image videos. If the batch size is 10, then as the train function reads in the data, it
            rearranges the tensors to be 20, 4 image videos.
        """

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            stacked images and its labels
        """


    def __len__(self):
        return len(self.)
