import os
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor
import torch
import numpy as np

from image_transformations import generate_pair


class DatasetPytorch(Dataset):
    
    def __init__(self, dataset, config, type_:str, apply_mg_transforms:bool):
        """[summary]

        Args:
            dataset ([type]): dataset.py class
            config ([type]): config.py class
            type_ (str): "train", "val" or "ts"
            apply_mg_transforms (bool) True for self supervision
        """
        
        self.dataset = dataset
        self.config = config
        self.type = type_
        self.apply_mg_transforms = apply_mg_transforms
        self.nr_samples_train = None
        self.nr_samples_val = None
        self.nr_samples_test = None
        
    def __len__(self):
        
        if self.type == "train":
            if self.nr_samples_train is None:
                np_array_names = self.dataset.x_train_filenames_original
                self.nr_samples_train = 0
                for f in np_array_names:
                    a = np.load(os.path.join(self.dataset.x_data_dir, f))
                    self.nr_samples_train += a.shape[0]
                return self.nr_samples_train
            else:
                return self.nr_samples_train
            
        if self.type == "val":
            if self.nr_samples_val is None:
                np_array_names = self.dataset.x_val_filenames_original
                self.nr_samples_val = 0
                for f in np_array_names:
                    a = np.load(os.path.join(self.dataset.x_data_dir, f))
                    self.nr_samples_val += a.shape[0]
                return self.nr_samples_val
            else:
                return self.nr_samples_val
            
        if self.type == "ts":
            if self.nr_samples_test is None:
                np_array_names = self.dataset.x_test_filenames_original
                self.nr_samples_test = 0
                for f in np_array_names:
                    a = np.load(os.path.join(self.dataset.x_data_dir, f))
                    self.nr_samples_test += a.shape[0]
                return self.nr_samples_test
            else:
                return self.nr_samples_test
    
    def __getitem__(self, idx):
        
        if self.type == "train":
            x, y = self.dataset.get_train(batch_size=1, return_tensor=True)
            if x is not None:
                if self.apply_mg_transforms:
                    x_transform, y = generate_pair(x, 1, self.config, make_tensors=True)
                    return (x_transform, y)
                return (x,y)
           
        if self.type == "val":
            x, y = self.dataset.get_val(batch_size=1, return_tensor=True)
            if x is not None:
                if self.apply_mg_transforms:
                    x_transform, y = generate_pair(x, 1, self.config, make_tensors=True)
                    return (x_transform, y)
                return (x, y)
        
        if self.type == "ts":
            x, y = self.dataset.get_test(batch_size=1, return_tensor=True)
            if x is not None:
                if self.apply_mg_transforms:
                    x_transform, y = generate_pair(x, 1, self.config, make_tensors=True)
                    return (x_transform, y)
                return (x, y)
        
    @staticmethod
    def custom_collate(batch):
        dims = batch[0][0].shape
        x_tensor = torch.zeros(len(batch), 1, dims[2], dims[3], dims[4])
        y_tensor = torch.zeros(len(batch), 1, dims[2], dims[3], dims[4])
        for idx, (x, y) in enumerate(batch): # y can be none in ss case
            x_tensor[idx] = x
            if y is None: y_tensor = None
            else: y_tensor[idx] = y
            
        return (x,y)
            
            
            
                