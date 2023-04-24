"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import os
from util.niqe import niqe
from util.piqe import piqe
import cv2

import torch.nn.functional as F

def synthesize(J, L):
    I = J * L
    return I

def reverse(I, L):

    J = I / L
    return torch.clamp(J,0,1)


def fuse_images(rec_J, refine_J):
    score_recpiqe = piqe(rec_J)
    score_refinepiqe = piqe(refine_J)
    score_recniqe = niqe(rec_J)
    score_refineniqe = niqe(refine_J)

    fuseWeightniqe = 1 - score_recniqe / (score_recniqe + score_refineniqe)
    fuseWeightpiqe = 1 - score_recpiqe / (score_recpiqe + score_refinepiqe)
    fuseWeight = (fuseWeightpiqe+fuseWeightniqe)/2

    return rec_J*fuseWeight + refine_J*(1-fuseWeight)

def get_tensor_dark_channel(img, neighborhood_size):
    shape = img.shape
    if len(shape) == 4:
        img_min = torch.min(img, dim=1)
        img_dark = F.max_pool2d(img_min, kernel_size=neighborhood_size, stride=1)
    else:
        raise NotImplementedError('get_tensor_dark_channel is only for 4-d tensor [N*C*H*W]')

    return img_dark



def array2Tensor(in_array, gpu_id=-1):
    in_shape = in_array.shape
    if len(in_shape) == 2:
        in_array = in_array[:,:,np.newaxis]

    arr_tmp = in_array.transpose([2,0,1])
    arr_tmp = arr_tmp[np.newaxis,:]

    if gpu_id >= 0:
        return torch.tensor(arr_tmp.astype(np.float)).to(gpu_id)
    else:
        return torch.tensor(arr_tmp.astype(np.float))


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def rescale_tensor(input_tensor):
    """"Converts a Tensor array into the Tensor array whose data are identical to the image's.
    [height, width] not [width, height]

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """

    if isinstance(input_tensor, torch.Tensor):
        input_tmp = input_tensor.cpu().float()
        output_tmp = input_tmp * 255.0
        output_tmp = output_tmp.to(torch.uint8)
    else:
        return input_tensor

    return output_tmp.to(torch.float32) / 255.0

    # if not isinstance(input_image, np.ndarray):
    #     if isinstance(input_image, torch.Tensor):  # get the data from a variable
    #         image_tensor = input_image.data
    #     else:
    #         return input_image
    #     image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
    #     image_numpy = (image_numpy + 1) / 2.0 * white_color  # post-processing: tranpose and scaling
    # else:  # if it is a numpy array, do nothing
    #     image_numpy = input_image
    # return torch.from_numpy(image_numpy)

def my_imresize(in_array, tar_size):
    oh = in_array.shape[0]
    ow = in_array.shape[1]

    if len(tar_size) == 2:
        h_ratio = tar_size[0]/oh
        w_ratio = tar_size[1]/ow
    elif len(tar_size) == 1:
        h_ratio = tar_size
        w_ratio = tar_size

    if len(in_array.shape) == 3:
        return ndimage.zoom(in_array, (h_ratio, w_ratio, 1), prefilter=False)
    else:
        return ndimage.zoom(in_array, (h_ratio, w_ratio), prefilter=False)

def psnr(img, ref, max_val=1):
    if isinstance(img, torch.Tensor):
        distImg = img.cpu().float().numpy()
    elif isinstance(img, np.ndarray):
        distImg = img.astype(np.float)
    else:
        distImg = np.array(img).astype(np.float)

    if isinstance(ref, torch.Tensor):
        refImg = ref.cpu().float().numpy()
    elif isinstance(ref, np.ndarray):
        refImg = ref.astype(np.float)
    else:
        refImg = np.array(ref).astype(np.float)

    rmse = np.sqrt( ((distImg-refImg)**2).mean() )
    # rmse = np.std(distImg-refImg) # keep the same with RESIDE's criterion
    return 20*np.log10(max_val/rmse)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
