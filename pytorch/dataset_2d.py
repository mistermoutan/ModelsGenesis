from numpy.random import shuffle
import numpy as np
from torch import Tensor

from dataset import Dataset


class Dataset2D:
    def __init__(self, dataset3d, limit_of_samples=100000):

        self.dataset3d = dataset3d
        # pointers
        self.x_filenames = self.dataset3d.x_filenames
        self.x_data_dir = self.dataset3d.x_data_dir
        self.x_train_filenames_original = self.dataset3d.x_train_filenames_original
        self.x_val_filenames_original = self.dataset3d.x_val_filenames_original
        self.x_test_filenames_original = self.dataset3d.x_test_filenames_original
        self.cube_dimensions = self.dataset3d.cube_dimensions
        self.tr_val_ts_split = self.dataset3d.tr_val_ts_split

        self.train_idxs, self.val_idxs = [[], [], []], [[], [], []]
        self.nr_samples_train = None
        self.nr_samples_val = None
        self.nr_samples_test = None
        self.allowed_slices_per_cube_train = None
        self.allowed_slices_per_cube_val = None
        print("INSTANCIATED DATASET 2D CLASS FOR {}".format(self.x_data_dir))
        self.train_sampling_idx, self.val_sampling_idx = 0, 0

        self.limit_of_samples = limit_of_samples
        self._calculate_nr_samples_train()
        self._calculate_nr_samples_val()
        self._set_cube_trimming()

    def get_train(self, batch_size: int, return_tensor=True) -> tuple():

        if sum([len(x) for x in self.train_idxs]) == 0:
            self.x_train_cube, self.y_train_cube = self.dataset3d.get_train(batch_size=1, return_tensor=False)  # (1,C,x,y,z)
            if self.x_train_cube is None:
                return (None, None)
            assert self.x_train_cube.shape[0] == 1 and self.x_train_cube.shape[1] == 1
            self.x_train_cube = self.x_train_cube[0]  # (1,C,x,y,z) -> (C,x,y,z)

            self.x_train_cube_x_view = np.transpose(self.x_train_cube, (1, 0, 2, -1))  # (C,x,y,z) -> (z,C,x,y)
            self.x_train_cube_y_view = np.transpose(self.x_train_cube, (2, 0, 1, -1))  # (C,x,y,z) -> (z,C,x,y)
            self.x_train_cube_z_view = np.transpose(self.x_train_cube, (-1, 0, 1, 2))  # (C,x,y,z) -> (z,C,x,y)
            self.x_train_cube_x_view, self.x_train_cube_y_view, self.x_train_cube_z_view = self._pad_if_necessary(
                [self.x_train_cube_x_view, self.x_train_cube_y_view, self.x_train_cube_z_view]
            )
            self.train_idxs = [
                [i for i in range(self.x_train_cube_x_view.shape[0])],
                [i for i in range(self.x_train_cube_y_view.shape[0])],
                [i for i in range(self.x_train_cube_z_view.shape[0])],
            ]

            if self.allowed_slices_per_cube_train is not None:
                self._trim_train_idxs()

            for idxs_list in self.train_idxs:
                shuffle(idxs_list)

            if self.y_train_cube is not None:
                self.y_train_cube = self.y_train_cube[0]  # (1,C,x,y,z) -> (C,x,y,z)
                self.y_train_cube_x_view = np.transpose(self.y_train_cube, (1, 0, 2, -1))  # (C,x,y,z) -> (z,C,x,y)
                self.y_train_cube_y_view = np.transpose(self.y_train_cube, (2, 0, 1, -1))  # (C,x,y,z) -> (z,C,x,y)
                self.y_train_cube_z_view = np.transpose(self.y_train_cube, (-1, 0, 1, 2))  # (C,x,y,z) -> (z,C,x,y)
                self.y_train_cube_x_view, self.y_train_cube_y_view, self.y_train_cube_z_view = self._pad_if_necessary(
                    [self.y_train_cube_x_view, self.y_train_cube_y_view, self.y_train_cube_z_view]
                )
        if self.train_sampling_idx == 0:
            x = self.x_train_cube_x_view[self.train_idxs[self.train_sampling_idx][:batch_size]]
        elif self.train_sampling_idx == 1:
            x = self.x_train_cube_y_view[self.train_idxs[self.train_sampling_idx][:batch_size]]
        elif self.train_sampling_idx == 2:
            x = self.x_train_cube_z_view[self.train_idxs[self.train_sampling_idx][:batch_size]]

        if return_tensor:
            x = Tensor(x)

        if self.dataset3d.has_target:
            if self.train_sampling_idx == 0:
                y = self.y_train_cube_x_view[self.train_idxs[self.train_sampling_idx][:batch_size]]
            elif self.train_sampling_idx == 1:
                y = self.y_train_cube_y_view[self.train_idxs[self.train_sampling_idx][:batch_size]]
            elif self.train_sampling_idx == 2:
                y = self.y_train_cube_z_view[self.train_idxs[self.train_sampling_idx][:batch_size]]

            if return_tensor:
                y = Tensor(y)
        else:
            assert self.y_train_cube is None
            y = None

        del self.train_idxs[self.train_sampling_idx][:batch_size]
        self._advance_index("train")
        return (x, y)

    def get_val(self, batch_size: int, return_tensor=True) -> tuple():

        if sum([len(x) for x in self.val_idxs]) == 0:
            self.x_val_cube, self.y_val_cube = self.dataset3d.get_train(batch_size=1, return_tensor=False)  # (1,C,x,y,z)
            if self.x_val_cube is None:
                return (None, None)

            assert self.x_val_cube.shape[0] == 1 and self.x_val_cube.shape[1] == 1
            self.x_val_cube = self.x_val_cube[0]  # (1,C,x,y,z) -> (C,x,y,z)

            self.x_val_cube_x_view = np.transpose(self.x_val_cube, (1, 0, 2, -1))  # (C,x,y,z) -> (z,C,x,y)
            self.x_val_cube_y_view = np.transpose(self.x_val_cube, (2, 0, 1, -1))  # (C,x,y,z) -> (z,C,x,y)
            self.x_val_cube_z_view = np.transpose(self.x_val_cube, (-1, 0, 1, 2))  # (C,x,y,z) -> (z,C,x,y)
            self.x_val_cube_x_view, self.x_val_cube_y_view, self.x_val_cube_z_view = self._pad_if_necessary(
                [self.x_val_cube_x_view, self.x_val_cube_y_view, self.x_val_cube_z_view]
            )
            self.val_idxs = [
                [i for i in range(self.x_val_cube_x_view.shape[0])],
                [i for i in range(self.x_val_cube_y_view.shape[0])],
                [i for i in range(self.x_val_cube_z_view.shape[0])],
            ]
            if self.allowed_slices_per_cube_val is not None:
                self._trim_val_idxs()

            for idxs_list in self.val_idxs:
                shuffle(idxs_list)

            if self.y_val_cube is not None:
                self.y_val_cube = self.y_val_cube[0]  # (1,C,x,y,z) -> (C,x,y,z)
                self.y_val_cube_x_view = np.transpose(self.y_val_cube, (1, 0, 2, -1))  # (C,x,y,z) -> (z,C,x,y)
                self.y_val_cube_y_view = np.transpose(self.y_val_cube, (2, 0, 1, -1))  # (C,x,y,z) -> (z,C,x,y)
                self.y_val_cube_z_view = np.transpose(self.y_val_cube, (-1, 0, 1, 2))  # (C,x,y,z) -> (z,C,x,y)
                self.y_val_cube_x_view, self.y_val_cube_y_view, self.y_val_cube_z_view = self._pad_if_necessary(
                    [self.y_val_cube_x_view, self.y_val_cube_y_view, self.y_val_cube_z_view]
                )
        if self.val_sampling_idx == 0:
            x = self.x_val_cube_x_view[self.val_idxs[self.val_sampling_idx][:batch_size]]
        elif self.val_sampling_idx == 1:
            x = self.x_val_cube_y_view[self.val_idxs[self.val_sampling_idx][:batch_size]]
        elif self.val_sampling_idx == 2:
            x = self.x_val_cube_z_view[self.val_idxs[self.val_sampling_idx][:batch_size]]

        if return_tensor:
            x = Tensor(x)

        if self.dataset3d.has_target:

            if self.val_sampling_idx == 0:
                y = self.y_val_cube_x_view[self.val_idxs[self.val_sampling_idx][:batch_size]]
            elif self.val_sampling_idx == 1:
                y = self.y_val_cube_y_view[self.val_idxs[self.val_sampling_idx][:batch_size]]
            elif self.val_sampling_idx == 2:
                y = self.y_val_cube_z_view[self.val_idxs[self.val_sampling_idx][:batch_size]]

            if return_tensor:
                y = Tensor(y)
        else:
            assert self.y_val_cube is None
            y = None

        del self.val_idxs[self.val_sampling_idx][:batch_size]
        self._advance_index("val")
        return (x, y)

    def get_len_train(self):

        self._calculate_nr_samples_train()
        if self.nr_samples_train < self.limit_of_samples:
            return self.nr_samples_train
        else:
            nr_cubes_train = self.dataset3d.get_len_train()
            return nr_cubes_train * self.allowed_slices_per_cube_train

    def get_len_val(self):

        self._calculate_nr_samples_val()
        if self.nr_samples_val < self.limit_of_samples:
            return self.nr_samples_val
        else:
            nr_cubes_val = self.dataset3d.get_len_val()
            return nr_cubes_val * self.allowed_slices_per_cube_val

    def reset(self):
        self.dataset3d.reset()
        # update references
        self.dataset3d.x_filenames = self.x_filenames
        self.dataset3d.x_train_filenames_original = self.x_train_filenames_original
        self.dataset3d.x_val_filenames_original = self.x_val_filenames_original
        self.dataset3d.x_test_filenames_original = self.x_test_filenames_original

    def _calculate_nr_samples_train(self):

        if self.nr_samples_train is None:
            self.nr_samples_train = 0
            nr_cubes_train = self.dataset3d.get_len_train()
            print("NR 3D CUBES TRAIN:", nr_cubes_train)
            cube_dims = self.dataset3d.get_cube_dimensions()
            nr_slices = sum(cube_dims)
            self.nr_samples_train = nr_cubes_train * nr_slices

    def _calculate_nr_samples_val(self):

        if self.nr_samples_val is None:
            nr_cubes_val = self.dataset3d.get_len_val()
            print("NR 3D CUBES VAL:", nr_cubes_val)
            cube_dims = self.dataset3d.get_cube_dimensions()
            nr_slices = sum(cube_dims)
            self.nr_samples_val = nr_cubes_val * nr_slices

    def _set_cube_trimming(self):

        if self.nr_samples_train > self.limit_of_samples:
            nr_cubes_train = self.dataset3d.get_len_train()
            self.allowed_slices_per_cube_train = int(np.ceil(self.limit_of_samples / nr_cubes_train))
            print("ALLOWED SLICES PER CUBE: ", self.allowed_slices_per_cube_train)

        if self.nr_samples_val > self.limit_of_samples:
            nr_cubes_val = self.dataset3d.get_len_val()
            self.allowed_slices_per_cube_val = int(np.ceil(self.limit_of_samples / nr_cubes_val))

    def _trim_train_idxs(self):

        len_idxs = [len(i) for i in self.train_idxs]
        total_idxs = sum(i for i in len_idxs)
        proportions = [i / total_idxs for i in len_idxs]

        # print("PRE TRIMMING", sum(len(i) for i in self.train_idxs))
        nr_to_trim = total_idxs - self.allowed_slices_per_cube_train  # how many slices must be deleted
        # print("NUMBER TO TRIM:", nr_to_trim)
        trim_each_view = [int(np.floor(i * nr_to_trim)) for i in proportions]
        while sum(i for i in trim_each_view) < nr_to_trim:
            trim_each_view[1] += 1
        for idx, nr_to_trim in enumerate(trim_each_view):
            del self.train_idxs[idx][:nr_to_trim]
        # print("POST TRIMMING", sum(len(i) for i in self.train_idxs))

    def _trim_val_idxs(self):

        len_idxs = [len(i) for i in self.val_idxs]
        total_idxs = sum(i for i in len_idxs)
        proportions = [i / total_idxs for i in len_idxs]

        # print("PRE TRIMMING", sum(len(i) for i in self.val_idxs))
        nr_to_trim = total_idxs - self.allowed_slices_per_cube_val  # how many slices must be deleted
        # print("NUMBER TO TRIM:", nr_to_trim)
        trim_each_view = [int(np.floor(i * nr_to_trim)) for i in proportions]
        while sum(i for i in trim_each_view) < nr_to_trim:
            trim_each_view[1] += 1
        for idx, nr_to_trim in enumerate(trim_each_view):
            del self.val_idxs[idx][:nr_to_trim]
        # print("POST TRIMMING", sum(len(i) for i in self.val_idxs))

    def _advance_index(self, type: str):

        if type == "train":
            if sum([len(x) for x in self.train_idxs]) == 0:
                return
            self.train_sampling_idx += 1
            try:
                self.train_idxs[self.train_sampling_idx]
            except IndexError:
                self.train_sampling_idx = 0
            if self.train_idxs[self.train_sampling_idx] == []:
                self._advance_index("train")

        if type == "val":
            if sum([len(x) for x in self.val_idxs]) == 0:
                return
            self.val_sampling_idx += 1
            try:
                self.val_idxs[self.val_sampling_idx]
            except IndexError:
                self.val_sampling_idx = 0
            if self.val_idxs[self.val_sampling_idx] == []:
                self._advance_index("val")

    @staticmethod
    def _pad_if_necessary(arrays: list) -> list:

        max_dim_x = max(arrays[0].shape[-2], arrays[1].shape[-2], arrays[2].shape[-2])
        max_dim_y = max(arrays[0].shape[-1], arrays[1].shape[-1], arrays[2].shape[-1])

        for idx, array in enumerate(arrays):
            pad = []
            if array.shape[-1] != max_dim_y:
                diff = max_dim_y - array.shape[-1]
                assert diff > 0
                if diff % 2 == 0:
                    pad.insert(0, (int(diff / 2), int(diff / 2)))
                else:
                    maior = int((diff - 1) / 2)
                    menor = int(diff - maior)
                    pad.insert(0, (maior, menor))
            else:
                pad.insert(0, (0, 0))

            if array.shape[-2] != max_dim_x:
                diff = max_dim_x - array.shape[-2]
                if diff % 2 == 0:
                    pad.insert(0, (int(diff / 2), int(diff / 2)))
                else:
                    maior = int((diff - 1) / 2)
                    menor = int(diff - maior)
                    pad.insert(0, (maior, menor))
            else:
                pad.insert(0, (0, 0))

            if pad != []:
                pad.insert(0, (0, 0))
                pad.insert(0, (0, 0))
                # print("PAD ", pad)
                arrays[idx] = np.pad(array, pad, "constant", constant_values=0)

        return arrays


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
