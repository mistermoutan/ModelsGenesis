from numpy.random import shuffle
import numpy as np
from torch import Tensor

from dataset import Dataset


class Dataset2D:
    def __init__(self, dataset3d):

        self.dataset3d = dataset3d
        # pointers
        self.x_filenames = self.dataset3d.x_filenames
        self.x_data_dir = self.dataset3d.x_data_dir
        self.x_train_filenames_original = self.dataset3d.x_train_filenames_original
        self.x_val_filenames_original = self.dataset3d.x_val_filenames_original
        self.x_test_filenames_original = self.dataset3d.x_test_filenames_original
        self.cube_dimensions = self.dataset3d.cube_dimensions
        self.tr_val_ts_split = self.dataset3d.tr_val_ts_split

        self.train_idxs, self.val_idxs = [], []
        self.nr_samples_train = None
        self.nr_samples_val = None
        self.nr_samples_test = None

    def get_train(self, batch_size: int, return_tensor=True) -> tuple():

        if not self.train_idxs:
            self.x_train_cube, self.y_train_cube = self.dataset3d.get_train(batch_size=1, return_tensor=False)  # (1,C,x,y,z)
            if self.x_train_cube is None:
                return (None, None)
            assert self.x_train_cube.shape[0] == 1 and self.x_train_cube.shape[1] == 1
            self.x_train_cube = self.x_train_cube[0]  # (1,C,x,y,z) -> (C,x,y,z)
            self.y_train_cube = self.y_train_cube[0]  # (1,C,x,y,z) -> (C,x,y,z)
            self.x_train_cube = np.transpose(self.x_train_cube, (-1, 0, 1, 2))  # (C,x,y,z) -> (z,C,x,y)
            self.y_train_cube = np.transpose(self.y_train_cube, (-1, 0, 1, 2))  # (C,x,y,z) -> (z,C,x,y)

            self.train_idxs = [i for i in range(self.x_train_cube.shape[0])]
            shuffle(self.train_idxs)

        x = self.x_train_cube[self.train_idxs[:batch_size]]
        if return_tensor:
            x = Tensor(x)
        if self.dataset3d.has_target:
            y = self.y_train_cube[self.train_idxs[:batch_size]]
            if return_tensor:
                y = Tensor(y)
        else:
            y = None

        del self.train_idxs[:batch_size]
        return (x, y)

    def get_val(self, batch_size: int, return_tensor=True) -> tuple():

        if not self.val_idxs:
            self.x_val_cube, self.y_val_cube = self.dataset3d.get_train(batch_size=1, return_tensor=False)  # (1,C,x,y,z)
            if self.x_val_cube is None:
                return (None, None)

            assert self.x_val_cube.shape[0] == 1 and self.x_val_cube.shape[1] == 1
            self.x_val_cube = self.x_val_cube[0]  # (1,C,x,y,z) -> (C,x,y,z)
            self.y_val_cube = self.y_val_cube[0]  # (1,C,x,y,z) -> (C,x,y,z)
            self.x_val_cube = np.transpose(self.x_val_cube, (-1, 0, 1, 2))  # (C,x,y,z) -> (z,C,x,y)
            self.y_val_cube = np.transpose(self.y_val_cube, (-1, 0, 1, 2))  # (C,x,y,z) -> (z,C,x,y)

            self.val_idxs = [i for i in range(self.x_val_cube.shape[0])]
            shuffle(self.train_idxs)

        x = self.x_val_cube[self.val_idxs[:batch_size]]
        if return_tensor:
            x = Tensor(x)
        if self.dataset3d.has_target:
            y = self.y_val_cube[self.val_idxs[:batch_size]]
            if return_tensor:
                y = Tensor(y)
        else:
            y = None

        del self.val_idxs[:batch_size]
        return (x, y)

    def get_len_train(self):
        # ATTENTION: May result in fuck up if nr_z_slices are not all the same for all cubes as I think happened in soem cube extractions
        if self.nr_samples_train is None:
            self.nr_samples_train = 0
            nr_cubes_train = self.dataset3d.get_len_train()
            nr_z_slices_per_cube = self.dataset3d.get_nr_z_slices_per_cube()
            self.nr_samples_train = nr_cubes_train * nr_z_slices_per_cube
        return self.nr_samples_train

    def get_len_val(self):
        # ATTENTION: May result in fuck up if nr_z_slices are not all the same for all cubes as I think happened in soem cube extractions
        if self.nr_samples_val is None:
            nr_cubes_val = self.dataset3d.get_len_val()
            nr_z_slices_per_cube = self.dataset3d.get_nr_z_slices_per_cube()
            self.nr_samples_val = nr_cubes_val * nr_z_slices_per_cube
        return self.nr_samples_val

    def reset(self):
        self.dataset3d.reset()
        # update references
        self.dataset3d.x_filenames = self.x_filenames
        self.dataset3d.x_train_filenames_original = self.x_train_filenames_original
        self.dataset3d.x_val_filenames_original = self.x_val_filenames_original
        self.dataset3d.x_test_filenames_original = self.x_test_filenames_original


if __name__ == "__main__":

    d3 = Dataset(data_dir="pytorch/datasets/heart_mri/datasets/x_cubes_full/extracted_cubes_64_64_12_sup", train_val_test=(0.8, 0.2, 0))
    d2 = Dataset2D(d3)
    print(d2.get_len_val())
    print(d2.get_len_train())
"""     cnt = 0
    while True:
        x, y = d2.get_train(batch_size=1)
        print(x.shape)
        if x is None:
            break
        # print(x.shape)
        cnt += x.shape[0]
    print(cnt) """
