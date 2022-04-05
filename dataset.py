import random
import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    # Makre sure the files are located in the /root/ directory
    # where:
    #   input (directory of scans subsampled by 2)
    #   target (directory of fully sampled data)
    #   val_input (directory of validation scans subsampled by 2)
    #   val_target (directory of validation scans fully sampled)


    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        if mode == "train": 
            self.files_input = sorted(glob.glob(os.path.join(root, "input") + "/*.*"))
            self.files_target = sorted(glob.glob(os.path.join(root, "target") + "/*.*"))
        elif mode == "val": 
            self.files_input = sorted(glob.glob(os.path.join(root, "val_input") + "/*.*"))
            self.files_target = sorted(glob.glob(os.path.join(root, "val_target") + "/*.*"))


    def __getitem__(self, index):

        #img = Image.open(self.files[index % len(self.files)])
        img_input = np.load(self.files_input[index % len(self.files_input)])
        img_target = np.load(self.files_target[index % len(self.files_target)])
        
        seed = np.random.randint(2147483647)
        
        random.seed(seed)
        torch.manual_seed(seed)
        img_input = self.transform(img_input)
        
        random.seed(seed)
        torch.manual_seed(seed)
        img_target = self.transform(img_target)

        return img_input, img_target

    def __len__(self):
        return len(self.files_input)