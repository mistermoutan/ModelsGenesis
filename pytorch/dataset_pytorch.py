import numpy as np
from torch.utils.data import Dataset as DatasetP
from torch import Tensor
import torch

from image_transformations import generate_pair
from dataset import Dataset
from dataset_2d import Dataset2D

from generic_supervision_transforms import TransformsForSupervision
from acs_paper_transforms import ACSPaperTransforms


class DatasetPytorch(DatasetP):
    def __init__(self, dataset, config, type_: str, apply_mg_transforms: bool):
        """[summary]

        Args:
            dataset ([type]): dataset.py class or dataset_2d.py class
            config ([type]): config class
            type_ (str): "train", "val" or "test"
            apply_mg_transforms (bool) True for self supervision
        """

        assert type_ in ("train", "val", "test")
        self.dataset = dataset
        self.config = config
        self.type = type_
        self.apply_mg_transforms = apply_mg_transforms
        self.nr_samples_used = 0

        if hasattr(self.dataset, "use_acs_paper_transforms") and self.dataset.use_acs_paper_transforms is True:
            print("GOING TO USE ACS PAPER TRANSFORMS")
            self.acs_transform_train = ACSPaperTransforms(size=48, move=5, train=True)
            self.acs_transform_test = ACSPaperTransforms(size=48, move=5, train=False)
            self.apply_acs_transforms = True
        else:
            self.apply_acs_transforms = False

        if hasattr(self.dataset, "use_supervision_transforms") and self.dataset.use_supervision_transforms is True:
            print("GOING TO USE SUPERVISION TRANSFORMS")
            self.supervison_transforms = TransformsForSupervision()
            self.apply_supervison_transforms = True
        else:
            self.apply_supervison_transforms = False

    def __len__(self):

        if self.type == "train":
            return self.dataset.get_len_train()

        if self.type == "val":
            return self.dataset.get_len_val()

        if self.type == "test":
            return self.dataset.get_len_test()

    def __getitem__(self, idx):

        if self.type == "train":
            x, y = (
                self.dataset.get_train(batch_size=1, return_tensor=False)
                if (self.apply_mg_transforms or self.apply_mg_transforms or self.apply_supervison_transforms)
                else self.dataset.get_train(batch_size=1, return_tensor=True)
            )
            if x is not None:
                self.nr_samples_used += 1
                self._check_reset()
                if self.apply_mg_transforms:
                    if isinstance(self.dataset, Dataset):
                        x_transform, y = generate_pair(x, 1, self.config, make_tensors=True)

                    elif isinstance(self.dataset, Dataset2D):
                        x_transform, y = generate_pair(x, 1, self.config, make_tensors=True, two_dim=True)
                    return (x_transform, y)

                elif self.apply_acs_transforms:
                    assert y is not None
                    if isinstance(self.dataset, Dataset):
                        x_transform, y_transform = self.acs_transform_train(x, y)
                        shape_x_transform = x_transform.shape
                        assert len(shape_x_transform) == 5 and shape_x_transform[0] == 1 and shape_x_transform[1] == 1
                    elif isinstance(self.dataset, Dataset2D):
                        raise NotImplementedError
                    return (Tensor(x_transform), Tensor(y_transform))

                elif self.apply_supervison_transforms:
                    assert y is not None
                    if isinstance(self.dataset, Dataset):
                        x_transform, y_transform = self.supervison_transforms(x, y)
                        shape_x_transform = x_transform.shape
                        assert len(shape_x_transform) == 5 and shape_x_transform[0] == 1 and shape_x_transform[1] == 1
                    elif isinstance(self.dataset, Dataset2D):
                        raise NotImplementedError
                    return (Tensor(x_transform), Tensor(y_transform))

                return (x, y)

        if self.type == "val":
            x, y = (
                self.dataset.get_val(batch_size=1, return_tensor=False)
                if (self.apply_mg_transforms or self.apply_acs_transforms or self.apply_supervison_transforms)
                else self.dataset.get_val(batch_size=1, return_tensor=True)
            )
            if x is not None:
                self.nr_samples_used += 1
                self._check_reset()
                if self.apply_mg_transforms:
                    if isinstance(self.dataset, Dataset):
                        x_transform, y = generate_pair(x, 1, self.config, make_tensors=True)
                    elif isinstance(self.dataset, Dataset2D):
                        x_transform, y = generate_pair(x, 1, self.config, make_tensors=True, two_dim=True)
                    return (x_transform, y)

                elif self.apply_acs_transforms:
                    assert y is not None
                    if isinstance(self.dataset, Dataset):
                        x_transform, y_transform = self.acs_transform_test(x, y)
                        shape_x_transform = x_transform.shape
                        assert len(shape_x_transform) == 5 and shape_x_transform[0] == 1 and shape_x_transform[1] == 1
                    elif isinstance(self.dataset, Dataset2D):
                        raise NotImplementedError
                    return (Tensor(x_transform), Tensor(y_transform))
                elif self.apply_supervison_transforms:
                    # no transforms for testing
                    return (Tensor(x), Tensor(y))
                return (x, y)

        if self.type == "test":
            x, y = (
                self.dataset.get_test(batch_size=1, return_tensor=False)
                if self.apply_mg_transforms
                else self.dataset.get_test(batch_size=1, return_tensor=True)
            )
            if x is not None:
                self.nr_samples_used += 1
                self._check_reset()
                if self.apply_mg_transforms:
                    x_transform, y = generate_pair(x, 1, self.config, make_tensors=True)
                    return (x_transform, y)
                return (x, y)

    def _check_reset(self):
        if self.nr_samples_used == self.__len__():
            self.nr_samples_used = 0
            self.reset()

    def reset(self):
        self.dataset.reset()

    """     def get_padding_to_make_cube_80_80_80_as_in_lidc(shape, wanted=(80,80,80)):
        # print("shape ", shape)
        # wanted = (80, 80, 80)
        slack = []
        for i, j in zip(wanted, shape):

            if i > j:
                slack.insert(0, (i - j))
            else:
                slack.insert(0, 0)

        pad = []
        for i in slack:
            if i % 2 == 0:
                pad.insert(0, (int(i / 2), int(i / 2)))
            else:
                maior = int((i - 1) / 2)
                menor = int(i - maior)
                pad.insert(0, (maior, menor))
        # print("padding ", pad)
        return pad
        
    @staticmethod
    def fix_dims_batch(batch):
        # in mini lidc cubes because not all have same dims
        shape_first_elem_of_batch = list(batch[0][0].shape)
        #get biggest
        for tuple_x_y in batch:
            for i, dim in enumerate(tuple_x_y[0].shape):
                if dim > shape_first_elem_of_batch[i]:
                    shape_first_elem_of_batch[i] = dim
        for tuple_x_y in batch:
            if list(tuple_x_y[0].shape) != shape_first_elem_of_batch:
                pass """

    @staticmethod
    def custom_collate(batch):
        dims = batch[0][0].shape
        if len(dims) == 5:
            x_tensor = torch.zeros(len(batch), 1, dims[2], dims[3], dims[4])
            y_tensor = torch.zeros(len(batch), 1, dims[2], dims[3], dims[4])
        elif len(dims) == 4:
            x_tensor = torch.zeros(len(batch), 1, dims[2], dims[3])
            y_tensor = torch.zeros(len(batch), 1, dims[2], dims[3])
        else:
            raise ValueError

        for idx, (x, y) in enumerate(batch):  # y can be none in ss case
            x_tensor[idx] = x
            if y is None:
                y_tensor = None
            else:
                y_tensor[idx] = y

        return (x_tensor, y_tensor)


if __name__ == "__main__":

    from config import models_genesis_config
    from torch.utils.data import DataLoader
    from dataset import Dataset

    # config = models_genesis_config()
    # x_train_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.train_fold]
    # x_val_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.valid_fold]
    # x_test_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.test_fold]  # Dont know in what sense they use this for
    # files = [x_train_filenames, x_val_filenames, x_test_filenames]
    # dataset = Dataset(config.data_dir, train_val_test=(0.8, 0.2, 0), file_names=files)  # train_val_test is non relevant as is overwritten by files

    config = models_genesis_config(False)
    x_train_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.train_fold]
    x_val_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.valid_fold]
    x_test_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.test_fold]  # Dont know in what sense they use this for
    files = [x_train_filenames, x_val_filenames, x_test_filenames]
    dataset_luna = Dataset(config.data_dir, train_val_test=(0.8, 0.2, 0), file_names=files)

    x_train_filenames = ["tr_cubes_64x64x32.npy"]
    x_val_filenames = ["val_cubes_64x64x32.npy"]
    x_test_filenames = ["ts_cubes_64x64x32.npy"]
    files = [x_train_filenames, x_val_filenames, x_test_filenames]
    dataset_lidc = Dataset(
        data_dir="pytorch/datasets/lidc_idri_cubes", train_val_test=(0.8, 0.2, 0), file_names=files
    )  # train_val_test is non relevant as is overwritte

    d1 = Dataset("pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes", (0.5, 0.3, 0.2))
    d2 = Dataset("pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes", (0.5, 0.5, 0))

    PD = DatasetPytorch(d1, config, type_="train", apply_mg_transforms=False)

    DL = DataLoader(PD, batch_size=6, num_workers=0, pin_memory=True)
    n_samples = PD.__len__()
    print(n_samples)
    while True:
        sample_count = 0
        print("new epoch")
        for iteration, (x, y) in enumerate(DL):
            sample_count += x.shape[0]
            if (iteration + 1) % 200 == 0:
                print(iteration, type(x), type(y))
            if sample_count == n_samples:
                print(sample_count, "/", n_samples)
                print("exhausted dataset")
            if sample_count > n_samples:
                print("OVERDOING")
        PD.reset()  # works
