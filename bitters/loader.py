from pathlib import Path
from random import randint, choice

import PIL

import torchvision.transforms.functional as F
import numpy as np
import numbers

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder, FakeData, VisionDataset
from pytorch_lightning import LightningDataModule
import torch
from typing import Any, Callable, Optional, Tuple
from torchvision import transforms

from PIL import Image

from bitters.modules.image_tokenizer.tokenizer import DiscreteImageTokenizer

from io import BytesIO

#To prevent truncated error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def identity(x):
    return x

class Grayscale2RGB:
    def __init__(self):  
        pass  
    def __call__(self, img):
        if img.mode != 'RGB':
            return img.convert('RGB') 
        else:
            return img
    def __repr__(self):
        return self.__class__.__name__ + '()'     

class ToTensor:
    def __call__(self, image):
        image = torch.as_tensor(image)
        return image

def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class SquarePad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, get_padding(img), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)   


class ImageDataset(Dataset):
    def __init__(self,
                 folder,
                 transform=None,
                 transform_dwt=None,
                 transform_tensor=None,
                 return_dwt = False,
                 shuffle=False,
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        self.return_dwt = return_dwt
        path = Path(folder)

        self.image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp'),
            *path.glob('**/*.PNG'), *path.glob('**/*.JPG'),
            *path.glob('**/*.JPEG'), *path.glob('**/*.BMP')            
        ]
        self.transform = transform
        self.transform_dwt = transform_dwt
        self.transform_tensor = transform_tensor

    def __len__(self):
        return len(self.image_files)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)
        
    def __getitem__(self, ind):
        try:
            image = self.transform(PIL.Image.open(self.image_files[ind]))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(corrupt_image_exceptions)            
            print(f"An exception occurred trying to load file {self.image_files[ind]}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)      

        if self.return_dwt:
            high = self.transform_dwt(image)
            mid = self.transform_dwt(high)
            low = self.transform_dwt(mid)         
            return self.transform_tensor(low), self.transform_tensor(mid), self.transform_tensor(high), self.transform_tensor(image)

        else:
            return self.transform_tensor(image)



class ImageDataModule(LightningDataModule):

    def __init__(self, train_dir, val_dir, batch_size, num_workers, img_size, return_dwt=False, resize_ratio=0.75):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.return_dwt = return_dwt
        self.transform_train = T.Compose([
                            Grayscale2RGB(),
                            T.RandomResizedCrop(img_size,
                                    scale=(resize_ratio, 1.),ratio=(1., 1.)),                      
                            ])
        self.transform_val = T.Compose([
                                    Grayscale2RGB(),
                                    T.Resize(img_size),
                                    T.CenterCrop(img_size),                              
                                    ])
        self.transform_dwt = T.Compose([
                                    DiscreteImageTokenizer(),                                                                   
                                    ])
        self.transform_tensor = T.Compose([                                   
                                    T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),                                
                                    ])
    def imagetransform(self, b):
        return Image.open(BytesIO(b))

    def dummy(self, s):
        return torch.zeros(1)

    def setup(self, stage=None):
        self.train_dataset = ImageDataset(self.train_dir, self.transform_train, self.transform_dwt, self.transform_tensor, self.return_dwt)
        self.val_dataset = ImageDataset(self.val_dir, self.transform_val, self.transform_dwt, self.transform_tensor, self.return_dwt)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

