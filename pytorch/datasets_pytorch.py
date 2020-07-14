from torch.utils.data import Dataset as DatasetP
from image_transformations import generate_pair


class DatasetsPytorch(DatasetP):
    def __init__(self, dataset, config, type_: str, apply_mg_transforms: bool):
        """[summary]

        Args:
            dataset ([type]): datasets.py class
            config ([type]): config class
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
