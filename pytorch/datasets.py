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
            self.nr_used_samples = 0 if self.max_samples_each else None

        # for key, value in kwargs.items():
        self.datasets = datasets
        self.nr_datasets = len(self.datasets)
        self.mode = mode
        self.stop_criteria_alternate = kwargs.get("stop_criteria_alternate", None)
        self.nr_exhausted_datasets = 0
        self.idx_train, self.idx_val, self.idx_test = 0, 0, 0

    def get_train(self, **kwargs):

        if self.mode == "alternate":
            dataset_to_sample = self.datasets[self.idx_train]
            x, y = dataset_to_sample.get_train(**kwargs)
            # x, y = dataset_to_sample.get_train(**kwargs["batch_size"], kwargs["return_tensor"])
            if x is None:  # dataset exhausted
                if self.stop_criteria_alternate == "first_exhausted":
                    return (None, None)
                elif self.stop_criteria_alternate == "all_exhausted":
                    self.nr_exhausted_datasets += 1
                    if self.nr_exhausted_datasets >= self.nr_datasets:
                        return (None, None)
                    del self.datasets[self.idx_train]
                    self._advance_index("train")
                    self.get_train(**kwargs)

            return (x, y)

        elif self.mode == "sequential":
            dataset_to_sample = self.datasets[self.idx_train]
            x, y = dataset_to_sample.get_train(**kwargs)

            if self.max_samples_each is not None:
                if self.nr_used_samples == self.max_samples_each:
                    self.nr_used_samples = 0
                    self.nr_exhausted_datasets += 1
                    self._advance_index("train")
                    print("exhausted", self.nr_exhausted_datasets, "datasets", " out of ", self.nr_datasets)
                if self.nr_exhausted_datasets >= self.nr_datasets:
                    return (None, None)
                self.nr_used_samples += 1
                return (x, y)

            elif x is None:
                self.nr_exhausted_datasets += 1
                if self.nr_exhausted_datasets >= self.nr_datasets:
                    return (None, None)
                self._advance_index("train")
                self.get_train()
            return (x, y)
        
    #TODO: IMPLEMENT get_val (Copy paste)
    
    def reset(self):
        for dataset in self.datasets:
            dataset.reset()

    def _advance_index(self, idx_to_adjust: str):

        if idx_to_adjust == "train":
            self.idx_train = 0 if self.idx_train == len(self.datasets) - 1 else self.idx_train + 1
        elif idx_to_adjust == "val":
            self.idx_val = 0 if self.idx_val == len(self.datasets) - 1 else self.idx_val + 1
        elif idx_to_adjust == "test":
            self.idx_test = 0 if self.idx_test == len(self.datasets) - 1 else self.idx_test + 1


if __name__ == "__main__":
    from dataset import Dataset

    d = Dataset("pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes", (0.5, 0.5, 0))
    d1 = Dataset("pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes", (0.5, 0.5, 0))
    lista = [d, d1]
    ds = Datasets(datasets=lista, mode="sequential", max_samples_each=2 )
    for idx, _ in enumerate(range(7)):
        x, y = ds.get_train(batch_size=4)
        print(type(x))
