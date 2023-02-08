import os.path
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import random
import cv2
from data.ace import zmIceColor
import numpy as np
import torchvision.transforms as transforms
import torch
'''from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True'''
class UnpairedDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_I = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_J = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'


        self.I_paths = sorted(make_dataset(self.dir_I, opt.max_dataset_size)) # load images from '/path/to/data/trainA'
        self.J_paths = sorted(make_dataset(self.dir_J, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.I_size = len(self.I_paths)  # get the size of dataset A
        self.J_size = len(self.J_paths)  # get the size of dataset B

    def __getitem__(self, index):

        I_path = self.I_paths[index % self.I_size]  # make sure index is within then range

        if self.opt.serial_batches:   # make sure index is within then range
            index_J = index % self.J_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_J = random.randint(0, self.J_size - 1)
        J_path = self.J_paths[index_J % self.J_size]

        I_img = Image.open(I_path).convert('RGB')
        J_img = Image.open(J_path).convert('RGB')

        params_I = get_params(self.opt, I_img.size)
        params_J = get_params(self.opt, J_img.size)

        transform_I = get_transform(self.opt, params=params_I, grayscale=(self.opt.input_nc == 1))
        transform_J = get_transform(self.opt, params=params_J, grayscale=(self.opt.output_nc == 1))

        # apply image transformation
        real_I = transform_I(I_img)
        real_J = transform_J(J_img)



        return {'dusty': real_I, 'clear': real_J, 'paths': I_path, 'J_paths': J_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.I_size, self.J_size)
