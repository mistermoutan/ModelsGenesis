from copy import deepcopy


class Datasets:
    """
    Sample from multiples datasets
    """

    def __init__(self, datasets: list, mode: str, **kwargs):
        """[summary]

        Arguments:
            datasets {list} -- [list cointaining instance of the single Dataset class]
            mode {str} -- [alternate: sample from D1, then from D2 ... ; sequential: EXHAUST samples from D1, then EXHAUST samples from D2 .. ]
            stop_criteria {str} -- [first_exhausted: once a D runs out of samples stop; all_exhausted: exhaust all D's]
        """

        assert mode == "alternate" or mode == "sequential"
        if mode == "alternate":
            assert kwargs.get("stop_criteria_alternate", None) == "first_exhausted" or "all_exhausted"
        elif mode == "sequential":
            self.max_samples_each = kwargs.get("max_samples_each", None)
            assert type(self.max_samples_each) == int or type(None)
            # ugly hack alarm: due to recursion if we want the same nr of samples for all, the first must start at 1
            self.nr_used_samples = 1 if self.max_samples_each else None

        self.datasets_tr = datasets
        self.datasets_val = deepcopy(datasets)
        self.datasets_test = deepcopy(dataset)
        self.datasets_copy = deepcopy(datasets)
        self.nr_datasets = len(self.datasets)
        self.mode = mode
        self.stop_criteria_alternate = kwargs.get("stop_criteria_alternate", None)
        self.nr_exhausted_datasets = 0
        self.idx_train, self.idx_val, self.idx_test = 0, 0, 0

    def get_train(self, **kwargs):

        if self.mode == "alternate":
            dataset_to_sample = self.datasets_tr[self.idx_train]
            # print("SAMPLING FROM ", self.idx_train)
            x, y = dataset_to_sample.get_train(**kwargs)
            # x, y = dataset_to_sample.get_train(**kwargs["batch_size"], kwargs["return_tensor"])
            if x is None:  # dataset exhausted
                if self.stop_criteria_alternate == "first_exhausted":  # CHECKED
                    # print("JUST EXHAUSTED DATASET ", self.idx_train)
                    return (None, None)
                elif self.stop_criteria_alternate == "all_exhausted":  # CHECKED
                    self.nr_exhausted_datasets += 1
                    # print("NUMBER EXHAUSTED DATASETS: ", self.nr_exhausted_datasets)
                    # print("NUMBER OF DATASETS:", len(self.datasets))
                    if self.nr_exhausted_datasets >= self.nr_datasets:
                        return (None, None)
                    del self.datasets_tr[self.idx_train]
                    # print("JUST DELETED DATASET ", self.idx_train)
                    # print("NUMBER OF DATASETS REMAINING:", len(self.datasets))
                    # print("PRE ADVANCE INDEX ", self.idx_train)
                    self._advance_index("train")
                    print("POST ADVANCE INDEX ", self.idx_train)

                    x, y = self.get_train(**kwargs)  #!!!
            self._advance_index("train")
            # print("FINAL INDEX", self.idx_train)
            return (x, y)

        elif self.mode == "sequential":
            dataset_to_sample = self.datasets_tr[self.idx_train]
            # print("TRYING TO SAMPLE FROM ", self.idx_train)
            x, y = dataset_to_sample.get_train(**kwargs)
            # if x is not None:
            # print("SAMPLING FROM ", self.idx_train)
            if self.max_samples_each is not None:  # CHECKED
                if self.nr_exhausted_datasets >= self.nr_datasets:
                    return (None, None)
                if x is None and self.nr_exhausted_datasets < self.nr_datasets:  # dataset exhausted and still some left
                    self.nr_used_samples = 0
                    self.nr_exhausted_datasets += 1
                    self._advance_index("train")
                    x, y = self.get_train(**kwargs)
                if x is None and self.nr_exhausted_datasets >= self.nr_datasets:
                    return (None, None)
                if self.nr_used_samples == self.max_samples_each:
                    # print("USED {} samples from dataset {} and is therefore exhausted".format(self.nr_used_samples,self.idx_train))
                    self.nr_used_samples = 0
                    self.nr_exhausted_datasets += 1
                    self._advance_index("train")
                    # print("exhausted", self.nr_exhausted_datasets, "datasets", " out of ", self.nr_datasets)
                    x, y = self.get_train(**kwargs)

                self.nr_used_samples += 1
                return (x, y)

            elif x is None:
                self.nr_exhausted_datasets += 1
                if self.nr_exhausted_datasets >= self.nr_datasets:
                    return (None, None)
                self._advance_index("train")
                x, y = self.get_train(**kwargs)
            return (x, y)

    def get_val(self, **kwargs):

        # print("TRYING TO SAMPLE FROM ", self.idx_val)
        dataset_to_sample = self.datasets_val[self.idx_val]
        x, y = dataset_to_sample.get_val(**kwargs)
        # if x is not None:
        #    print("SAMPLING FROM ", self.idx_val)
        if x is None:
            if self.idx_val + 1 >= self.nr_datasets:  # idx starts at 0 boy
                return (None, None)
            self._advance_index("val")
            x, y = self.get_val(**kwargs)

        return (x, y)

    def get_test(self, **kwargs):

        dataset_to_sample = self.datasets_test[self.idx_test]
        x, y = dataset_to_sample.get_test(**kwargs)
        if x is None:
            if self.idx_val + 1 >= self.nr_datasets:  # idx starts at 0 boy
                return (None, None)
            self._advance_index("test")
            x, y = self.get_val(**kwargs)

        return (x, y)

    def reset(self):
        self.datasets_tr = deepcopy(self.datasets_copy)
        self.datasets_val = deepcopy(self.datasets_copy)
        self.datasets_test = deepcopy(self.datasets_copy)
        self.nr_exhausted_datasets = 0
        self.idx_train, self.idx_val, self.idx_test = 0, 0, 0
        if hasattr(self, "nr_used_samples"):
            self.nr_used_samples = 0

        # not necessary as operating on copy of original array
        # for dataset in self.datasets:
        #    dataset.reset()

    def _advance_index(self, idx_to_adjust: str):

        if idx_to_adjust == "train":
            self.idx_train = 0 if self.idx_train >= len(self.datasets_tr) - 1 else self.idx_train + 1
        elif idx_to_adjust == "val":
            self.idx_val = 0 if self.idx_val >= len(self.datasets_val) - 1 else self.idx_val + 1
        elif idx_to_adjust == "test":
            self.idx_test = 0 if self.idx_test >= len(self.datasets_test) - 1 else self.idx_test + 1

    @property
    def cube_dimensions(self):
        return self.datasets_copy[0].cube_dimensions


if __name__ == "__main__":
    from dataset import Dataset
    from config import models_genesis_config

    # d = Dataset("pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes", (0.5, 0.5, 0))
    # ds = Datasets(datasets=[d], mode="sequential", max_samples_each=2 )
    # print(isinstance(ds,Datasets))
    config = models_genesis_config()
    dataset = Dataset(config.data_dir, train_val_test=(0.8, 0.2, 0))  # train_val_test is non relevant as will ve overwritten after
    dataset.x_train_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.train_fold]
    dataset.x_val_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.valid_fold]
    dataset.x_test_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.test_fold]  # Dont know in what sense they use this fo

    d = Dataset("pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes", (0.5, 0.5, 0))
    d1 = Dataset("pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes", (0.5, 0.5, 0))
    d2 = Dataset("pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes", (0.5, 0.5, 0))
    del d1.x_train_filenames[1:]

    lista = [d, dataset, d1, d2]
    ds = Datasets(datasets=lista, mode="sequential", stop_criteria_alternate="first_exhausted")
    cnt = 0
    import random

    random.seed(1)
    for _ in range(2):
        print("\n Iteration", _)
        while True:
            x, y = ds.get_val(batch_size=100)
            if x is None:
                print("BROKE")
                break
            print(x.shape, type(y))
            cnt += 1
        ds.reset()
