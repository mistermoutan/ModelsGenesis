import os
import numpy as np
import pandas as pd
from numpy.testing import *


def split_cubes_and_target_seg_into_separate_files(data_path="pytorch/datasets/lidc_acs_provided"):

    cubes_save_dir = os.path.join(data_path, "x")
    seg_save_dir = os.path.join(data_path, "y")
    os.makedirs(cubes_save_dir, exist_ok=True)
    os.makedirs(seg_save_dir, exist_ok=True)
    names = os.listdir(os.path.join(data_path, "nodule"))

    for name in names:
        with np.load(os.path.join(data_path, "nodule", name)) as npz:
            cube, seg = npz["voxel"], npz["answer1"]
            assert_raises(AssertionError, assert_array_equal, cube, seg)
            assert seg.any()  # not all zeros
            cube = cube / 255.0 - 1  # already normalize cubes

        np.save(os.path.join(cubes_save_dir, name[:-4]), np.expand_dims(cube, 0))
        np.save(os.path.join(seg_save_dir, name)[:-4], np.expand_dims(seg, 0))


if __name__ == "__main__":
    split_cubes_and_target_seg_into_separate_files()