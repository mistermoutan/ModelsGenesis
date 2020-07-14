from os import listdir, path
import os
from random import shuffle, sample
from math import ceil, floor
from copy import deepcopy

from torch import Tensor
import numpy as np


class Dataset:

    # TODO: remove self.has_target logic

    def __init__(self, data_dir: str, train_val_test: tuple, file_names=None):
        """
        Arguments:
            data_dir {str} -- [folder must be organized with x/ and y/ folders inside with .npy files, in y folder files must be named as the ones in x but ending in _target.npy]
            train_val_test {tuple} -- [proportion of train,validation and test examples] (default: {(0.65,0.15,0.20)})
            
        Keyword Arguments:
            file_names {[[], [], []]} -- Pre specify file names used for tr, val and test. Will override train_val_test split behavior
            
        """
        self.x_data_dir = path.join(data_dir, "x/")  # dir has an x and y folder
        self.y_data_dir = path.join(data_dir, "y/")
        self.has_target = listdir(self.y_data_dir) != []
        self.x_filenames = listdir(self.x_data_dir)
        shuffle(self.x_filenames)
        self.tr_val_ts_split = train_val_test
        if file_names is None:
            self.x_train_filenames, self.x_val_filenames, self.x_test_filenames = self.do_file_split(self.x_filenames, train_val_test)
        else:
            self.x_train_filenames, self.x_val_filenames, self.x_test_filenames = file_names[0], file_names[1], file_names[2]

        self.x_train_filenames_original, self.x_val_filenames_original, self.x_test_filenames_original = deepcopy(self.x_train_filenames), deepcopy(self.x_val_filenames), deepcopy(self.x_test_filenames)
        self.train_idxs, self.val_idxs, self.test_idxs = [], [], []
        self.cube_dimensions = deepcopy(self.x_train_filenames[0][-14:-6])
        self.reseted = False
        self.nr_samples_train = None
        self.nr_samples_val = None
        self.nr_samples_test = None

    def _load_data(self, tr_vl_ts_prop: tuple, force_load=(False, False, False)):
        """
        Arguments:
            tr_vl_ts_prop {tuple} -- tuple of 3 Bools, choose which type of data to load (training, validation, testing)

        Keyword Arguments:
            force_load {tuple} -- force loading the next volume (default: {(False,False,False)}) , used for ignoring batch sizes which do not conform to specified value
        """

        # only one volume will be in memory at a time
        # if all cubes from current volume were used and there are still volumes left to load
        if (not self.train_idxs and self.x_train_filenames and tr_vl_ts_prop[0]) or force_load[0]:
            x_train_file_name = self.x_train_filenames[0]
            # print("SAMPLING FROM", x_train_file_name)
            del self.x_train_filenames[0]
            self.x_array_tr = np.expand_dims(np.load(path.join(self.x_data_dir, x_train_file_name)), axis=1)  # (N, x, y, z) -> (N, 1-Channel, x, y, z)
            if self.has_target:
                self.y_array_tr = np.expand_dims(np.load(path.join(self.y_data_dir, x_train_file_name[:-4] + "_target.npy")), axis=1)
                assert self.x_array_tr.shape == self.y_array_tr.shape
            self.train_idxs = [i for i in range(self.x_array_tr.shape[0])]
            shuffle(self.train_idxs)

        if (not self.val_idxs and self.x_val_filenames and tr_vl_ts_prop[1]) or force_load[1]:

            x_val_file_name = self.x_val_filenames[0]
            del self.x_val_filenames[0]
            self.x_array_val = np.expand_dims(np.load(path.join(self.x_data_dir, x_val_file_name)), axis=1)  # (N, x, y, z) -> (N, 1, x, y, z)
            if self.has_target:
                self.y_array_val = np.expand_dims(np.load(path.join(self.y_data_dir, x_val_file_name[:-4] + "_target.npy")), axis=1)
                assert self.x_array_val.shape == self.y_array_val.shape
            self.val_idxs = [i for i in range(self.x_array_val.shape[0])]
            shuffle(self.val_idxs)

        if (not self.test_idxs and self.x_test_filenames and tr_vl_ts_prop[2]) or force_load[2]:

            x_test_file_name = self.x_test_filenames[0]
            del self.x_test_filenames[0]
            self.x_array_test = np.expand_dims(np.load(path.join(self.x_data_dir, x_test_file_name)), axis=1)  # (N, x, y, z) -> (N, 1, x, y, z)
            if self.has_target:
                self.y_array_test = np.expand_dims(np.load(path.join(self.y_data_dir, x_test_file_name[:-4] + "_target.npy")), axis=1)
                assert self.x_array_test.shape == self.y_array_test.shape
            self.test_idxs = [i for i in range(self.x_array_test.shape[0])]
            shuffle(self.test_idxs)

    def get_train(self, batch_size: int, return_tensor=True) -> tuple():
        """
        Returns: tuple(Tensor, Tensor) or tuple(None,None) if all examples have been exhausted
        """

        self.reseted = False
        self._load_data((True, False, False))
        if self.has_target:
            x, y = self.x_array_tr[self.train_idxs[:batch_size]], self.y_array_tr[self.train_idxs[:batch_size]]  # (batch_size ,1, x, y, z)
            assert x.shape == y.shape
            del self.train_idxs[:batch_size]
            if return_tensor:
                return (Tensor(x), Tensor(y)) if x.shape[0] != 0 else (None, None)
            else:
                return (x, y) if x.shape[0] != 0 else (None, None)
        else:
            x = self.x_array_tr[self.train_idxs[:batch_size]]  # (batch_size ,1, x, y, z)
            del self.train_idxs[:batch_size]
            if return_tensor:
                return (Tensor(x), None) if x.shape[0] != 0 else (None, None)
            else:
                return (x, None) if x.shape[0] != 0 else (None, None)

        # in case we can not accept smaller batches
        # if x.shape[0] != batch_size:
        #   self._load_data((True,False,False), force_load=(True,False,False))
        #   self.get_train(batch_size)

    def get_val(self, batch_size: int, return_tensor=True) -> tuple():

        self.reseted = False
        self._load_data((False, True, False))
        if self.has_target:
            x, y = self.x_array_val[self.val_idxs[:batch_size]], self.y_array_val[self.val_idxs[:batch_size]]  # (batch_size ,1, x, y, z)
            assert x.shape == y.shape
            del self.val_idxs[:batch_size]
            if return_tensor:
                return (Tensor(x), Tensor(y)) if x.shape[0] != 0 else (None, None)
            else:
                return (x, y) if x.shape[0] != 0 else (None, None)
        else:
            x = self.x_array_val[self.val_idxs[:batch_size]]  # (batch_size ,1, x, y, z)
            del self.val_idxs[:batch_size]
            if return_tensor:
                return (Tensor(x), None) if x.shape[0] != 0 else (None, None)
            else:
                return (x, None) if x.shape[0] != 0 else (None, None)

    def get_test(self, batch_size: int, return_tensor=True) -> tuple():

        self.reseted = False
        self._load_data((False, False, True))
        if self.has_target:
            x, y = self.x_array_test[self.test_idxs[:batch_size]], self.y_array_test[self.test_idxs[:batch_size]]  # (batch_size, 1, x , y, z)
            assert x.shape == y.shape
            del self.test_idxs[:batch_size]
            if return_tensor:
                return (Tensor(x), Tensor(y)) if x.shape[0] != 0 else (None, None)
            else:
                return (x, y) if x.shape[0] != 0 else (None, None)
        else:
            x = self.x_array_test[self.test_idxs[:batch_size]]  # (batch_size, 1, x , y, z)
            del self.test_idxs[:batch_size]
            if return_tensor:
                return (Tensor(x), None) if x.shape[0] != 0 else (None, None)
            else:
                return (x, None) if x.shape[0] != 0 else (None, None)

    def reset(self):
        # after epoch necessary because list will be empty so will return None when fecthing data
        self.x_train_filenames = deepcopy(self.x_train_filenames_original)
        self.x_val_filenames = deepcopy(self.x_val_filenames_original)
        self.x_test_filenames = deepcopy(self.x_test_filenames_original)
        self.reseted = True

    def get_len_train(self):

        if self.nr_samples_train is None:
            self.nr_samples_train = 0
            for f in self.x_train_filenames_original:
                a = np.load(os.path.join(self.x_data_dir, f))
                self.nr_samples_train += a.shape[0]
            return self.nr_samples_train
        else:
            return self.nr_samples_train

    def get_len_val(self):

        if self.nr_samples_val is None:
            self.nr_samples_val = 0
            for f in self.x_val_filenames_original:
                a = np.load(os.path.join(self.x_data_dir, f))
                self.nr_samples_val += a.shape[0]
            return self.nr_samples_val
        else:
            return self.nr_samples_val

    def get_len_test(self):

        if self.nr_samples_test is None:
            self.nr_samples_test = 0
            for f in self.x_test_filenames_original:
                a = np.load(os.path.join(self.x_data_dir, f))
                self.nr_samples_test += a.shape[0]
            return self.nr_samples_test
        else:
            return self.nr_samples_test

    @staticmethod
    def do_file_split(file_names: list, proportions: tuple) -> ([], [], []):

        train_prop, val_prop, _ = proportions
        assert train_prop + val_prop + _ == float(1)
        i = ceil(train_prop * len(file_names))
        j = ceil(len(file_names) * val_prop)
        assert len(file_names[:i]) + len(file_names[i : i + j]) + len(file_names[i + j :]) == len(file_names)
        return file_names[:i], file_names[i : i + j], file_names[i + j :]


if __name__ == "__main__":

    a = Dataset(data_dir="pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes", train_val_test=(0.1, 0, 0.9))
    print(a.x_train_filenames)
    for epoch in range(100):
        x = 0
        while x is not None:
            x, y = a.get_train(batch_size=20)
            if type(x) == type(None):
                print("X IS NONE")
            else:
                print(x.shape)
        a.reset()
        print("epoch {} done".format(epoch))
