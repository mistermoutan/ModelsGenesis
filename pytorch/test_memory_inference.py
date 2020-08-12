import torch
import SimpleITK as sitk
import numpy as np
from time import time

from unet3d import UNet3D

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

with torch.no_grad():
    unet3d.eval()
    pred = unet3d(img_array)
    print(pred.shape)
