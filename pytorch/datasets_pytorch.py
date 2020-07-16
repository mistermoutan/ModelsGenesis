from torch.utils.data import Dataset as DatasetP
from image_transformations import generate_pair
import torch


class DatasetsPytorch(DatasetP):
    def __init__(self, datasets: list, type_: str, mode: str, batch_size: int, apply_mg_transforms: bool):
        """[summary]

        Args:
            dataset ([type]): list of DatasetsPytorch / Dataset? 
            config ([type]): config class
            type_ (str): "train", "val" or "ts"
            apply_mg_transforms (bool) True for self supervision
        """

        self.datasets = datasets
        self.batch_size = batch_size

        assert mode in ("alternate", "sequential")
        self.mode = mode
        if self.mode == "alternate":
            self.exhausted_datasets = [False for i in range(len(self.datasets))]

        for dataset in self.datasets:
            assert dataset.type == type_
            assert dataset.apply_mg_transforms == apply_mg_transforms

        self.nr_samples = None
        self.sampling_idx = 0
        self.samples_used = [0 for i in range(len(self.datasets))]
        self.len_datasets = [len(d) for d in self.datasets]

    def __len__(self):

        if self.nr_samples is None:
            self.nr_samples = 0
            for d in self.datasets:
                self.nr_samples += d.__len__()
            return self.nr_samples
        return self.nr_samples

    def __getitem__(self, idx):

        # dataset_to_sample_from = self.datasets[self.sampling_idx]
        # samples_used = self.samples_used[self.sampling_idx]

        # for i, j in zip(self.samples_used, self.len_datasets):
        #    if i == j:
        # print(self.samples_used)
        #        break

        if self.mode == "sequential":
            len_dataset_to_sample_from = len(self.datasets[self.sampling_idx])
            if self.samples_used[self.sampling_idx] < len_dataset_to_sample_from:
                return_tuple = self.datasets[self.sampling_idx].__getitem__(idx) + (self.sampling_idx,)
                self.samples_used[self.sampling_idx] += 1
                if self.samples_used[self.sampling_idx] == len_dataset_to_sample_from:
                    self._advance_index()
            else:
                raise RuntimeError("Can't Get HERE SEQUENTIAL")

            self._check_reset()
            return return_tuple

        if self.mode == "alternate":

            while self.exhausted_datasets[self.sampling_idx] is True:
                self._advance_index()

            len_dataset_to_sample_from = len(self.datasets[self.sampling_idx])

            if self.samples_used[self.sampling_idx] < len_dataset_to_sample_from:
                return_tuple = self.datasets[self.sampling_idx].__getitem__(idx) + (self.sampling_idx,)
                self.samples_used[self.sampling_idx] += 1

                if (sum(self.samples_used) != 0) and (sum(self.samples_used) % self.batch_size) == 0:
                    self._advance_index()

                # shitty fix, ideally: if self.samples_used[self.sampling_idx] == len(self.datasets[self.sampling_idx]):
                for i in range(len(self.datasets)):
                    if self.samples_used[i] == len(self.datasets[i]):
                        self.exhausted_datasets[i] = True

            else:
                print("SHOULD NOT GET HERE ALTERNATE SAMPLING")
                self._advance_index()

            self._check_reset()
            return return_tuple

    def _advance_index(self):

        self.sampling_idx += 1
        try:
            self.datasets[self.sampling_idx]
        except IndexError:
            self.sampling_idx = 0

    def _check_reset(self):
        # automatic reset, MP safety check, do it automatically in instance as you can't access it
        cnt = 0
        for i, j in zip(self.samples_used, self.len_datasets):
            if i < j:
                break
            cnt += 1

        if cnt == len(self.datasets):
            self.reset()

    def reset(self):

        # print("DATASETS IS THE REASON")
        # Can only get here when num workers <= 1 if you need to increase this
        # will just need to add a check to which dataset you're sampling from in sequentail mode I think
        for idx in range(len(self.datasets)):
            self.datasets[idx].reset()

        self.sampling_idx = 0
        self.samples_used = [0 for i in range(len(self.datasets))]
        if self.mode == "alternate":
            self.exhausted_datasets = [False for i in range(len(self.datasets))]

    @staticmethod
    def custom_collate(batch):

        dims = batch[0][0].shape
        x_tensor = torch.zeros(len(batch), 1, dims[2], dims[3], dims[4])
        y_tensor = torch.zeros(len(batch), 1, dims[2], dims[3], dims[4])
        dataset_idxs = []
        for idx, (x, y, dataset_idx) in enumerate(batch):  # y can be none in ss case

            dataset_idxs.append(dataset_idx)
            x_tensor[idx] = x
            if y is None:
                y_tensor = None
            else:
                y_tensor[idx] = y

        # purify batches
        if len(set(dataset_idxs)) != 1:
            # print("PURIFYING BATCH")
            # print(dataset_idxs)
            cut_off_idx = 0
            for idx, i in enumerate(dataset_idxs):
                if idx == 0:
                    current_idx = i
                else:
                    if i != current_idx:
                        cut_off_idx = i
                        break
                    else:
                        current_idx = i

            # print("CUT OFF IDX ", cut_off_idx)
            # print("shape before ", x_tensor.shape)
            x_tensor = x_tensor[:cut_off_idx]
            # print("shape after", x_tensor.shape)
            if y_tensor is not None:
                y_tensor = y_tensor[:cut_off_idx]

        return (x_tensor, y_tensor)


if __name__ == "__main__":
    import torch

    from config import models_genesis_config
    from torch.utils.data import DataLoader
    from dataset import Dataset
    from datasets import Datasets
    from dataset_pytorch import DatasetPytorch

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
    dataset_lidc = Dataset(data_dir="pytorch/datasets/lidc_idri_cubes", train_val_test=(0.8, 0.2, 0), file_names=files)  # train_val_test is non relevant as is overwritte

    d1 = Dataset("pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes", (0.5, 0.3, 0.2))
    d2 = Dataset("pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes", (0.5, 0.5, 0))

    lista = [d1, dataset_lidc, d2]
    for i in range(len(lista)):
        lista[i] = DatasetPytorch(lista[i], config, type_="train", apply_mg_transforms=False)
    print(lista)

    num_workers = 2

    PDS = DatasetsPytorch(lista, type_="train", mode="alternate", batch_size=4, apply_mg_transforms=False)
    DL = DataLoader(PDS, batch_size=4, num_workers=num_workers, collate_fn=DatasetsPytorch.custom_collate, pin_memory=True)

    n_samples = PDS.__len__()
    sample_count = 0
    print(n_samples)
    # exit(0)

    while True:
        print("new epoch")
        sample_count = 0
        for iteration, (x, y) in enumerate(DL):
            print(PDS.samples_used)
            # print(PDS.len_datasets)
            sample_count += x.shape[0]
            # print(x.shape)
            if sample_count >= n_samples:
                print("exhausted dataset")
                print(PDS.__len__(), sample_count)

        PDS.reset()

