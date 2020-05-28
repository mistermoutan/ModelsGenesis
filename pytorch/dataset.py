from os import listdir, path

from random import shuffle, sample
from math import ceil, floor

from torch import Tensor
import numpy as np

class Dataset():

    def __init__(self, data_dir:str, train_val_test=(0.65,0.15,0.20)):
        """
        Arguments:
            data_dir -- [folder must be organized with x/ and y/ folders inside with .npy files, in y folder files must be named as the ones in x but ending in _target.npy]
            
        Keyword Arguments:
            train_val_test {tuple} -- [proportion of train,validation and test examples] (default: {(0.65,0.15,0.20)})
        """
        
        self.x_data_dir = path.join(data_dir, "x/") #dir has an x and y folder
        self.y_data_dir = path.join(data_dir, "y/")
        x_filenames = listdir(self.x_data_dir)
        shuffle(x_filenames)
        self.tr_val_ts_split = train_val_test
        self.x_train_filenames, self.x_val_filenames, self.x_test_filenames = self.do_file_split(x_filenames, train_val_test)
        self.train_idxs, self.val_idxs, self.test_idxs = [] , [] , []
        
    def _load_data(self, tr_vl_ts_prop:tuple, force_load=(False,False,False)):
        """
        Arguments:
            tr_vl_ts_prop {tuple} -- tuple of 3 Bools, choose which type of data to load (training, validation, testing)

        Keyword Arguments:
            force_load {tuple} -- force loading the next volume (default: {(False,False,False)}) , used for ignoring batch sizes which do not conform to specified value
        """
    
        # only one volume will be in memory at a time
        #if all cubes from current volume were used and there are still volumes left to load
        if (not self.train_idxs and self.x_train_filenames and tr_vl_ts_prop[0]) or force_load[0]: 
            
            x_train_file_name = self.x_train_filenames[0]
            print("LOADED")
            del self.x_train_filenames[0]
            self.x_array_tr = np.expand_dims(np.load(path.join(self.x_data_dir, x_train_file_name)), axis = 1) # (N, x, y, z) -> (N, 1, x, y, z)
            self.y_array_tr = np.expand_dims(np.load(path.join(self.y_data_dir, x_train_file_name[:-4] + "_target.npy")), axis=1)
            assert self.x_array_tr.shape == self.y_array_tr.shape
            self.train_idxs = [i for i in range(self.x_array_tr.shape[0])] 
            shuffle(self.train_idxs)
    
        if (not self.val_idxs and self.x_val_filenames and tr_vl_ts_prop[1]) or force_load[1]:
            
            x_val_file_name = self.x_val_filenames[0]
            del self.x_val_filenames[0]
            self.x_array_val = np.expand_dims(np.load(path.join(self.x_data_dir, x_val_file_name)), axis=1) # (N, x, y, z) -> (N, 1, x, y, z)
            self.y_array_val = np.expand_dims(np.load(path.join(self.y_data_dir, x_val_file_name[:-4] + "_target.npy")), axis=1)
            assert self.x_array_val.shape == self.y_array_val.shape
            self.val_idxs = [i for i in range(self.x_array_val.shape[0])]
            shuffle(self.val_idxs)
            
        if (not self.test_idxs and self.x_test_filenames and tr_vl_ts_prop[2]) or force_load[2]:
            
            x_test_file_name = self.x_test_filenames[0]
            del self.x_test_filenames[0]
            self.x_array_test = np.expand_dims(np.load(path.join(self.x_data_dir, x_test_file_name)), axis=1) # (N, x, y, z) -> (N, 1, x, y, z)
            self.y_array_test = np.expand_dims(np.load(path.join(self.y_data_dir, x_test_file_name[:-4] + "_target.npy")), axis=1)
            assert self.x_array_test.shape == self.y_array_test.shape
            self.test_idxs = [i for i in range(self.x_array_test.shape[0])] 
            shuffle(self.test_idxs)
        
    def get_train(self, batch_size:int) -> tuple():
        """
        Returns: tuple(Tensor, Tensor) or tuple(None,None) if all examples have been exhausted
        """
        
        self._load_data((True,False,False))
        x , y = self.x_array_tr[self.train_idxs[:batch_size]], self.y_array_tr[self.train_idxs[:batch_size]] # (batch_size ,1, x, y, z)
        assert x.shape == y.shape          
        del self.train_idxs[:batch_size]
        # in case we can not accept smaller batches
        # if x.shape[0] != batch_size:
        #   self._load_data((True,False,False), force_load=(True,False,False))
        #   self.get_train(batch_size)
        return (Tensor(x), Tensor(y)) if x.shape[0] != 0 else (None,None) 
    
    def get_val(self, batch_size:int) -> tuple():
        
        self._load_data((False,True,False))
        x , y = self.x_array_val[self.val_idxs[:batch_size]], self.y_array_val[self.val_idxs[:batch_size]] # (batch_size ,1, x, y, z)
        assert x.shape == y.shape          
        del self.val_idxs[:batch_size]
        return (Tensor(x), Tensor(y)) if x.shape[0] != 0 else (None,None)
    
    def get_test(self, batch_size:int) -> tuple():
    
        self._load_data((False, False, True))
        x , y = self.x_array_test[self.test_idxs[:batch_size]], self.y_array_test[self.test_idxs[:batch_size]] # (batch_size, 1, x , y, z)
        assert x.shape == y.shape          
        del self.test_idxs[:batch_size]
        return (Tensor(x), Tensor(y)) if x.shape[0] != 0 else (None,None)
    
    @staticmethod
    def do_file_split(file_names:list, proportions:tuple) -> ([],[],[]):
        
        train_prop, val_prop, _ = proportions
        i = ceil(train_prop * len(file_names)) 
        j = ceil((len(file_names) - i) * val_prop)
        return file_names[:i] , file_names[i:i+j], file_names[i+j:]
        
         