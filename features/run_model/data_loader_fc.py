import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import glob
from numpy import array
import numpy as np
import torch

import preprocessing as pre_process

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
        
        video_dir= []
        labels = []
        vid_range_start = []
        vid_range_end = []
        
        # Read in the image list (tab delimited) line by line 
        with open(image_list_file, "r") as f:
            file = iter(f)
            next(file)
            for line in file:
                
                items = line.strip().split('\t')
                
                # Get video name (folder holding images)
                video_name= items[0]
                
                # Get label. This code is specialized for Anueploid or Euploid.
                label = items[2]
                
                # Get subtype (no_feta_heart_no_sac) (Anuepoid has none)
                subtype = items[3]
                
                vid_range_start = items[7]
                vid_range_end = items[8]
                print(data_dir)
                print(label)
                print(subtype)
                print(video_name)
                print(data_dir + label + "_Zip_Files/"+ subtype + "/" + video_name)
                # Deal with the differences between Aneu and Eu -- subtypes-- and assign numeric labels
                if label == "Aneuploid":
                    #set 2 videos names are combined with their embryo #
                    if subtype == "set2":
                        vid_split_1 = video_name.split("_")
                        vid_number = vid_split_1[2].split("#")
                        new_vid_name = vid_split_1[0] + vid_number[1] + "_" + vid_split_1[1] + "_" + vid_split_1[2]
                        print(new_vid_name)
                        lab = 0
                        sub = glob.glob(data_dir + label + "_Zip_Files/"+ subtype + "/" + new_vid_name + '*')
                        print(sub)
                        sub_dir = sub[0] 
                        video_dir.append(sub_dir)
                    if subtype != "set2":
                        lab = 0
                        sub = glob.glob(data_dir + label + "_Zip_Files/"+ subtype + "/" + video_name + '*')
                        print(sub)
                        sub_dir = sub[0] 
                        video_dir.append(sub_dir) 
                else:
                    if subtype == "set2":
                        vid_split_1 = video_name.split("_")
                        vid_number = vid_split_1[2].split("#")
                        new_vid_name = vid_split_1[0] + vid_number[1] + "_" + vid_split_1[1] + "_" + vid_split_1[2]
                        print(new_vid_name)
                        lab = 0
                        sub = glob.glob(data_dir + label + "_Zip_Files/"+ subtype + "_zip_files/" + new_vid_name + '*')
                        print(sub)
                        sub_dir = sub[0] 
                        video_dir.append(sub_dir)
                    if subtype != "set2":
                        lab = 1
                        sub = glob.glob(data_dir + label + "_Zip_Files/"+ subtype + "_zip_files/" + video_name + '*')
                        print(sub)
                        sub_dir = sub[0]
                        video_dir.append(sub_dir)
     
                # Save values in lists
                labels.append(int(lab))

        self.video_dir = video_dir
        self.labels = labels
        self.time_tile = time_tile
        self.vid_range_start = vid_range_start
        self.vid_range_end = vid_range_end
        
    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            stacked images and its labels
        """
        
        current_video = self.video_dir[index]
        current_label = self.labels[index]
        time_tile = self.time_tile
        vid_range_start = self.vid_range_start
        vid_range_end = self.vid_range_end
                
        current_video_F0 = current_video + "/F0/"  
        #current_video_F15 = current_video + "/F15/"
        #current_video_F30 = current_video + "/F30/"
        #current_video_F45 = current_video + "/F45/"        
        #current_video_Fminus15 = current_video + "/F-15/"
        #current_video_Fminus30 = current_video + "/F-30/"
        #current_video_Fminus45 = current_video + "/F-45/"    
        
        #current_video_subdir_list = [current_video_F0, current_video_F15,
        #                             current_video_F30, current_video_F45,
        #                             current_video_Fminus15, current_video_Fminus30,
        #                             current_video_Fminus45]
        
        current_video_subdir_list = [current_video_F0]
        
        save_videos_list = []
        
        sample = {}

        # Specify function to be called on each list element
        def key_func(x):
            return os.path.split(x)[-1]
        
        # Get ranges of the video 
        stop = int(vid_range_end)
        begin = int(vid_range_start) 
        
        for j in current_video_subdir_list:
            # Get file names that match the specifications and sort (so they will be in order)
            current_image = sorted(glob.glob(j + '*' + "RUN"+ '*' + ".JPG"), key=key_func)
            
            # Read in the images for that video within the ranges determined above
            arrays_in = [array(Image.open(y)) for y in current_image[begin:stop]]

            arrays = list(map(pre_process.image_preprocessing, arrays_in))

            # Calculate the size of the tile
            tile_size = len(arrays)/int(time_tile)

            if int(time_tile) == 1:
                save_vid = np.stack(arrays[0:int(tile_size)])
                save_vid = torch.from_numpy(save_vid)
                save_vid = save_vid.type(torch.FloatTensor)
                save_vid = save_vid.unsqueeze(0)
            else:
                save_vid = np.stack(arrays[0:int(tile_size)])
                save_vid = torch.from_numpy(save_vid)
                save_vid = save_vid.type(torch.FloatTensor)
                for i in range(1,int(time_tile)):
                    # stack the first X images to create a subset video -- tiled along time            
                    vid = np.stack(arrays[int(i*tile_size):int(tile_size*(i+1))])
                    # convert to torch
                    vid = torch.from_numpy(vid)
                    # convert to float tensor
                    vid = vid.type(torch.FloatTensor)
                    # stack the tiles onto the stacked tiles
                    save_vid = torch.stack((save_vid, vid), dim = 0)
                    
            save_videos_list.append(save_vid)
            print(len(save_videos_list))
        save = torch.stack(save_videos_list,1)
        print("initial_shape")
        print(save.shape)
        save = save.view(-1, int(time_tile), int(1*tile_size), 500, 500)    
        print("second_shape")
        print(save.shape)
        save = torch.squeeze(save,1)    
        print(save.shape)
            
        sample.update({'vid': save, 'label': current_label})
        return sample

    def __len__(self):
        return len(self.video_dir)
