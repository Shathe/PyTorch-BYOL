from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

import glob


class STL(Dataset):   #Tranform-> leer con opencv.. to tensor
    def __init__(self, paths, transform=None):
        super().__init__()
        self.imgs = []
        if isinstance(paths, list):
            for path in paths:
                self.imgs = self.imgs + glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.jpg'))
        else:
            self.imgs = glob.glob(os.path.join(paths, '*.png')) + glob.glob(os.path.join(paths, '*.jpg'))

        self.path = paths
        self.trans = transform

        print(f'{len(self.imgs)} images found')
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        return self.trans(img),   0 # TODO: change the labels
