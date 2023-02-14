import numbers
import bitters.modules.image_tokenizer.dwt as dwt

import numpy as np
from PIL import Image

class DiscreteImageTokenizer(object):
    def __init__(self, img_size=256):
        assert isinstance(img_size, numbers.Number)
        self.img_size = img_size

    def get_seq_len(self):
        return (self.img_size //16) **2

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be encoded.

        Returns:
            np.array: numpy array of encoded vectors
        """
        return self.encode(img)
    
    def __repr__(self):
        return self.__class__.__name__ + '(img_size={0})'.\
            format(self.img_size)  

    def normalize(self, img):
        img = (img - img.min())/(img.max() - img.min() + 1e-8)
        return img           

    def encode(self, img):
        img = np.array(img)
        dwt_r, dwt_g, dwt_b = dwt.extract_coeff(img)
        img = np.stack((dwt_r,dwt_g, dwt_b), axis = 2)           

        img = self.normalize(img) * 255.0
        img = np.clip(img, 0, 255)
        return np.uint8(img)

tokenizer = DiscreteImageTokenizer()
