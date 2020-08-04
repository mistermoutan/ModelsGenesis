from copy import deepcopy
from math import ceil
from itertools import combinations
from sklearn.model_selection import KFold

from utils import *
from dataset import Dataset
from dataset_2d import Dataset2D

# FOR Deterministic Behavior this should be a dataset folder sort of component that is just read from and not a class generated
# at each experimetn level


class CrossValidator:
    def __init__(self, config, dataset, nr_splits=5):

        self.config = config
        self.dataset = dataset
        self.nr_splits = nr_splits
        self._generate_splits()
        # self.datasets_previous_runs = self._get_datasets_previous_runs()

    def set_dataset(self, d):
        self.dataset = d

    def set_config(self, c):
        self.config = c

    def override_dataset_files_with_splits(self):

        if isinstance(self.dataset, Dataset) or isinstance(self.dataset, Dataset2D):
            train_split, val_split = self.splits[self.dataset.x_data_dir].pop()
            self.dataset.x_train_filenames_original = train_split
            self.dataset.x_val_filenames_original = val_split
            self.dataset.x_test_filenames_original = []
            self.dataset.reset()
            print(
                "OVERWROTE DATASET FILES WITH CROSS VALIDATOR // {} SPLITS LEFT FOR {}".format(
                    len(self.splits[self.dataset.x_data_dir]), self.dataset.x_data_dir
                )
            )
        elif isinstance(self.dataset, list):
            for d in self.dataset:
                assert isinstance(d, Dataset)
                train_split, val_split = self.splits[d.dataset.x_data_dir].pop()
                d.x_train_filenames_original = train_split
                d.dataset.x_val_filenames_origianl = val_split
                d.x_test_filenames_original = []
                d.dataset.reset()
                print(
                    "OVERWROTE DATASET FILES WITH CROSS VALIDATOR // {} SPLITS LEFT FOR {}".format(
                        len(self.splits[d.dataset.x_data_dir]), d.dataset.x_data_dir
                    )
                )

    def _generate_splits(self):

        current_run_nr = self.config.experiment_nr
        if hasattr(self, "splits"):
            # this doesnt happen anyway as its only instanciated once and then loaded on further uses
            return
        if current_run_nr == 1:
            self.splits = dict()  # {dataset_x_dir : [ (train_split1, val_split1), (train_split2, val_split2) ... ] }
            if isinstance(self.dataset, Dataset) or isinstance(self.dataset, Dataset2D):
                self.dataset.x_filenames.sort()
                dataset_splits = self._generate_splits_from_filenames(self.dataset.x_filenames)
                self.splits[self.dataset.x_data_dir] = dataset_splits

            elif isinstance(self.dataset, list):
                for d in self.dataset:
                    assert isinstance(d, Dataset)
                    d.x_filenames.sort()
                    dataset_splits = self._generate_splits_from_filenames(d.x_filenames)
                    self.splits[d.x_data_dir] = dataset_splits

            else:
                raise ValueError("dataset must be Dataset or [Dataset's]")

    def _generate_splits_from_filenames(self, dataset_filenames):

        dataset_splits = []
        kf = KFold(n_splits=self.nr_splits, random_state=1, shuffle=True)
        for train_split, val_split in kf.split(dataset_filenames):
            train_filenames = [dataset_filenames[i] for i in train_split]
            val_filenames = [dataset_filenames[i] for i in val_split]
            dataset_splits.append((train_filenames, val_filenames))
        return dataset_splits


if __name__ == "__main__":
    dataset_splits = []
    kf = KFold(n_splits=3, random_state=1)
    for train_split, val_split in kf.split([i for i in range(5)]):
        print(train_split, val_split)
