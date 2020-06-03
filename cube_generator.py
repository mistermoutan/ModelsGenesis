import os
import sys
import random
from tqdm import tqdm
from optparse import OptionParser
from skimage.transform import resize
import SimpleITK as sitk
import numpy as np

from pytorch.utils import make_dir

"""
python cube_generator.py --data "pytorch/datasets/Task02_Heart/imagesTr/" --modality "mri" --scale 100 --target_dir "pytorch/datasets/Task02_Heart/labelsTr/" 
"""

sys.setrecursionlimit(40000)
parser = OptionParser()

parser.add_option("--input_rows", dest="input_rows", help="size of x dimension of the cube", default=64, type="int")
parser.add_option("--input_cols", dest="input_cols", help="size of y dimension of the cube", default=64, type="int")
parser.add_option("--input_deps", dest="input_deps", help="size of z dimension of the cube", default=32, type="int")
parser.add_option("--crop_rows", dest="crop_rows", help="crop rows", default=64, type="int")
parser.add_option("--crop_cols", dest="crop_cols", help="crop cols", default=64, type="int")
parser.add_option("--data", dest="data", help="directory of dataset", default=None, type="string")
#parser.add_option("--save", dest="save", help="the output directory of processed 3D cubes, if non existent will be created", default=None, type="string")
parser.add_option("--scale", dest="scale", help="number of cubes extracted from a single volume", default=32, type="int")
parser.add_option("--modality", dest="modality", help="ct or mri", default="None", type="string")
parser.add_option("--target_dir", dest="target_dir", help="target volume dir for label generation", default=None ,type="string")

(options, args) = parser.parse_args()


seed = 1
random.seed(seed)

assert options.data is not None
assert options.modality is not None, "input --modality , ct or mri"



class setup_config():
    """
    If the image modality in your target task is CT, we suggest that all the intensity values be clipped on the min (-1000) and max (+1000) 
    interesting Hounsfield Unitrange and then scale between 0 and 1. If the image modality is MRI, we suggest that all 
    the intensity values be clipped on min 0) and max (+4000) interesting range and then scale between 0 and 1.
     For any other modalities, you may want to first clip on the meaningful intensity range and then scale between 0 and 1.
    """
    def __init__(self, 
                 input_rows=None, 
                 input_cols=None,
                 input_deps=None,
                 crop_rows=None,
                 crop_cols=None,
                 len_border=None,
                 len_border_z=None,
                 scale=None,
                 DATA_DIR=None,
                 len_depth=None,
                 modality=None,
                 target_dir=None
                ):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_deps = input_deps
        self.crop_rows = crop_rows
        self.crop_cols = crop_cols
        self.len_border = len_border
        self.len_border_z = len_border_z
        self.scale = scale
        self.DATA_DIR = DATA_DIR
        self.len_depth = len_depth
        self.hu_max = 1000.0 if modality == "ct" else 4000
        self.hu_min = -1000.0 if modality =="mri" else 0
        self.target_dir = target_dir
        
            
            
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")



config = setup_config(input_rows=options.input_rows,
                      input_cols=options.input_cols,
                      input_deps=options.input_deps,
                      crop_rows=options.crop_rows,
                      crop_cols=options.crop_cols,
                      scale=options.scale,
                      len_border=100,
                      len_border_z=30,
                      len_depth=    3,
                      DATA_DIR = options.data,
                      target_dir = options.target_dir
                     )
config.display()

def infinite_generator_from_one_volume(config, img_array, target_array=None):
    
    size_x, size_y, size_z = img_array.shape
    
    if size_z-config.input_deps-config.len_depth-1-config.len_border_z < config.len_border_z:
        return None
    
    #min-max normalization
    img_array[img_array < config.hu_min] = config.hu_min
    img_array[img_array > config.hu_max] = config.hu_max
    img_array = 1.0*(img_array-config.hu_min) / (config.hu_max-config.hu_min)
    slice_set = np.zeros((config.scale, config.input_rows, config.input_cols, config.input_deps), dtype=float)
    slice_set_target = np.zeros((config.scale, config.input_rows, config.input_cols, config.input_deps), dtype=float)
    
    num_pair = 0
    cnt = 0
    
    while True:
        cnt += 1
        if cnt > 50 * config.scale and num_pair == 0:
            return None
        elif cnt > 50 * config.scale and num_pair > 0:
            return np.array(slice_set[:num_pair])

        start_x = random.randint(0+config.len_border, size_x-config.crop_rows-1-config.len_border)
        start_y = random.randint(0+config.len_border, size_y-config.crop_cols-1-config.len_border)
        start_z = random.randint(0+config.len_border_z, size_z-config.input_deps-config.len_depth-1-config.len_border_z)
        
        #get the cube
        crop_window = img_array[start_x : start_x+config.crop_rows,
                                start_y : start_y+config.crop_cols,
                                start_z : start_z+config.input_deps+config.len_depth,
                               ]

        if target_array is not None:
            assert type(target_array) == np.ndarray
            crop_window_target = target_array[start_x : start_x+config.crop_rows,
                                start_y : start_y+config.crop_cols,
                                start_z : start_z+config.input_deps+config.len_depth,
                               ]

        if config.crop_rows != config.input_rows or config.crop_cols != config.input_cols:
            crop_window = resize(crop_window, 
                                 (config.input_rows, config.input_cols, config.input_deps+config.len_depth), 
                                 preserve_range=True,
                                )
            if target_array:
                crop_window_target = resize (crop_window_target, 
                                 (config.input_rows, config.input_cols, config.input_deps+config.len_depth), 
                                 preserve_range=True,
                                )
        
        slice_set[num_pair] = crop_window[:,:,:config.input_deps]
        if target_array is not None:
            slice_set_target[num_pair] = crop_window_target[:,:,:config.input_deps]

        
        num_pair += 1
        if num_pair == config.scale:
            break
    
    if target_array is None:
        return np.array(slice_set)
    else:
        return np.array(slice_set), np.array(slice_set_target)

def get_self_learning_data(config):

    cubes_dir = os.path.join(config.DATA_DIR, "extracted_cubes") #where to save cubes
    make_dir(os.path.join(cubes_dir, "x/"))
    make_dir(os.path.join(cubes_dir, "y/"))
    
    volumes_file_names = [i for i in os.listdir(config.DATA_DIR) if "." != i[0] and i.endswith(".nii.gz")] 
    
    for volume in tqdm(volumes_file_names):

        itk_img = sitk.ReadImage(os.path.join(config.DATA_DIR, volume))
        img_array = sitk.GetArrayFromImage(itk_img)
        img_array = img_array.transpose(2, 1, 0)
        
        if not config.target_dir:
            x = infinite_generator_from_one_volume(config, img_array)
            if x is not None:
                print("Saving cubes of volume {}.  Dimensions: {} | {:.2f} ~ {:.2f}".format(volume, x.shape, np.min(x), np.max(x)))
                np.save(os.path.join(cubes_dir,
                            volume + "_" + 
                            str(config.scale)+
                            "_"+str(config.input_rows)+
                            "x"+str(config.input_cols)+
                            "x"+str(config.input_deps)+
                            ".npy"),
                    x)
        else:
            itk_img_tr = sitk.ReadImage(os.path.join(config.target_dir, volume))
            img_array_tr = sitk.GetArrayFromImage(itk_img_tr)
            img_array_tr = img_array_tr.transpose(2, 1, 0)
            assert img_array_tr.shape == img_array.shape
            res = infinite_generator_from_one_volume(config,img_array, target_array=img_array_tr)
            if res is not None:
                x, y = res
                if y is None or x is None:
                    assert x is None or y is None
                
                print("Saving cubes of volume {}.  Dimensions: {} {} | {:.2f} ~ {:.2f}".format(volume, x.shape, y.shape, np.min(x), np.max(x)))
                

                np.save(os.path.join(cubes_dir, "x", 
                            volume + "_" +
                            str(config.scale)+
                            "_"+str(config.input_rows)+
                            "x"+str(config.input_cols)+
                            "x"+str(config.input_deps)+
                            ".npy"),
                    x)
                np.save(os.path.join(cubes_dir, "y",
                            volume + "_" +
                            str(config.scale)+
                            "_"+str(config.input_rows)+
                            "x"+str(config.input_cols)+
                            "x"+str(config.input_deps)+
                            "_target.npy"),
                    y)                    

get_self_learning_data(config)
