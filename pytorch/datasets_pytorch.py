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

        self.break_time = False  # DATALOADER IS NOT EXITING BY ITSELF

        self.blah = 10
        self.wait = False

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

        if self.mode == "sequential":
            len_dataset_to_sample_from = len(self.datasets[self.sampling_idx])
            if self.samples_used[self.sampling_idx] < len_dataset_to_sample_from:
                i = 0
                return_list = []
                while i < self.batch_size and self.samples_used[self.sampling_idx] < len_dataset_to_sample_from:
                    return_list.append(self.datasets[self.sampling_idx].__getitem__(idx))
                    self.samples_used[self.sampling_idx] += 1
                    i += 1

                if self.samples_used[self.sampling_idx] == len_dataset_to_sample_from:
                    self._advance_index()

            # elif self.samples_used[self.sampling_idx] == len_dataset_to_sample_from:
            #    self._advance_index()
            #    return_list = self.__getitem__(idx)
            else:
                raise RuntimeError("Can't Get HERE")

            return return_list

        if self.mode == "alternate":

            while self.exhausted_datasets[self.sampling_idx] is True:
                self._advance_index()

            len_dataset_to_sample_from = len(self.datasets[self.sampling_idx])
            if self.samples_used[self.sampling_idx] < len_dataset_to_sample_from:
                i = 0
                return_list = []
                while i < self.batch_size and self.samples_used[self.sampling_idx] < len_dataset_to_sample_from:
                    return_list.append(self.datasets[self.sampling_idx].__getitem__(idx))
                    self.samples_used[self.sampling_idx] += 1
                    i += 1
                if self.samples_used[self.sampling_idx] == len_dataset_to_sample_from:
                    self.exhausted_datasets[self.sampling_idx] = True

                self._advance_index()

            else:
                raise RuntimeError("Can't Get HERE")

            return return_list

    def _advance_index(self):

        cnt = 0
        for i, j in zip(self.samples_used, self.len_datasets):
            if i < j:
                break
            cnt += 1

        if cnt == len(self.datasets):
            self.break_time = True
            self.reset()
        # if self.samples_used == self.len_datasets:
        #    self.break_time = True

        self.sampling_idx += 1
        try:
            self.datasets[self.sampling_idx]
        except IndexError:
            self.sampling_idx = 0
        # if self.sampling_idx < len(self.datasets) - 1:
        #    self.sampling_idx += 1
        # else:
        #    self.sampling_idx = 0

        # rint("SELF:SAMPLES USED", self.samples_used)
        # rint([len(d) for d in self.datasets])
        # print(self.exhausted_datasets, "\n")

    def reset(self):

        for idx in range(len(self.datasets)):
            self.datasets[idx].reset()

        self.sampling_idx = 0
        self.samples_used = [0 for i in range(len(self.datasets))]
        if self.mode == "alternate":
            self.exhausted_datasets = [False for i in range(len(self.datasets))]

        # MULTI PROCSSING SCREWED THIS UP
        # if self.break_time is not True:
        #    raise RuntimeError("SHOULD BE TRUE to call reset")
        self.break_time = False

    @staticmethod
    def custom_collate(batch):

        import torch

        dims = batch[0][0][0].shape
        x_tensor = torch.zeros(len(batch[0]), 1, dims[2], dims[3], dims[4])
        y_tensor = torch.zeros(len(batch[0]), 1, dims[2], dims[3], dims[4])
        for idx, (x, y) in enumerate(batch[0]):  # y can be none in ss case
            x_tensor[idx] = x
            if y is None:
                y_tensor = None
            else:
                y_tensor[idx] = y

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

    lista = [d1]
    for i in range(len(lista)):
        lista[i] = DatasetPytorch(lista[i], config, type_="train", apply_mg_transforms=True)
    print(lista)

    num_workers = 3
    PDS = DatasetsPytorch(lista, type_="train", mode="sequential", batch_size=config.batch_size_ss, apply_mg_transforms=True)
    DL = DataLoader(PDS, batch_size=1, num_workers=num_workers, collate_fn=DatasetsPytorch.custom_collate, pin_memory=True)

    n_samples = PDS.__len__()
    sample_count = 0
    print(n_samples)

    while True:
        print("new epoch")
        sample_count = 0
        for iteration, (x, y) in enumerate(DL):
            sample_count += x.shape[0]
            if sample_count >= n_samples:
                print("exhausted dataset")
                print(PDS.__len__(), sample_count)
            if sample_count >= len(PDS):
                print("BREAK TIME")
                DL = DataLoader(PDS, batch_size=1, num_workers=num_workers, collate_fn=DatasetsPytorch.custom_collate, pin_memory=True)
                break
            print(sample_count)
        # PDS.reset()

