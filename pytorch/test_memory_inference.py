import torch
import SimpleITK as sitk
import numpy as np
from time import time
from utils import *
from unet3d import UNet3D
from random import choice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet3d = UNet3D()
unet3d.to(device)

itk_img = sitk.ReadImage("/work1/s182312/medical_decathlon/Task02_Heart/imagesTr/la_011.nii.gz")
img_array = sitk.GetArrayFromImage(itk_img)
img_array = img_array.transpose(2, 1, 0)
img_array = np.expand_dims(img_array, (0, 1))

img_array = torch.Tensor(img_array)
img_array = img_array.float().to(device)
img_array = img_array.contiguous()
img_array, _ = pad_if_necessary_one_array(img_array, return_pad_tuple=False)


def do_inference(array):

    with torch.no_grad():
        unet3d.eval()
        try:
            pred = unet3d(array)
            return pred.shape
        except:
            return None


def clip_array(array):
    possible = [idx for idx, i in enumerate(array.shape) if i > 16]
    dim = choice(possible)
    if dim == 2:
        array = array[:, :, 15:, :, :]
    elif dim == 3:
        array = array[:, :, :, 15:, :]
    elif dim == 4:
        array = array[:, :, :, :, 15:]
    else:
        raise ValueError

    return array


while do_inference(img_array) is None:
    img_array = clip_array(img_array)

print("WORKED: ", img_array.shape)
s = do_inference(img_array)
print(s)
