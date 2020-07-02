import glob
import random
import os
import numpy as np
import torch
from scipy import io

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets

class ImageDataset(Dataset):
    def __init__(self,
                 transforms_=None,
                 mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files = None
        #self.files = sorted(glob.glob('%s/*.jpg' % root))
        #self.files = self.files[:-2000] if mode == 'train' else self.files[-2000:]

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        filename = filepath.split('/')[-1]
        img = self.transform(Image.open(filepath).convert('RGB'))
        return img

    def __len__(self):
        return len(self.files)

class CelebADataset(ImageDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files = glob.glob('%s/*.jpg' % root)

class CarsDataset(ImageDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files = glob.glob('%s/cars_test/*.jpg' % root) + \
            glob.glob("%s/cars_train/*.jpg" % root)
