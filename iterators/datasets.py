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

class CelebADataset(Dataset):
    def __init__(self,
                 root,
                 transforms_=None,
                 mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob('%s/*.jpg' % root))
        self.files = self.files[:-2000] if mode == 'train' else self.files[-2000:]

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        filename = filepath.split('/')[-1]
        img = self.transform(Image.open(filepath).convert('RGB'))
        return img

    def __len__(self):
        return len(self.files)
