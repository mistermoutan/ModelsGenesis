import os
import dill


def make_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_object(o, name, dir_):

    directory = os.path.join(dir_, "{}.pkl".format(name))
    with open(directory, "wb") as f:
        dill.dump(o, f)
    print("SAVED {} object in {}".format(name, dir_))


def load_object(location):

    with open(location, "rb") as f:
        o = dill.load(f)
    print("LOADED {} object".format(location))
    return o


def replace_obj_attributes(config_object, **kwargs):

    for key, value in kwargs.items():
        if not hasattr(config_object, key):
            raise AttributeError("Config does not have this attribute")
        assert type(value) == type(config_object.key), "Trying to replace a {} attribute by a {}".format(str(type(config_object.key)), str(type(value)))
        config_object.key = value


dataset_map = {
    "lidc": "/work1/s182312/lidc_idri/np_cubes",
}


def build_dataset(dataset_list: list, split: tuple, mode: str):

    from dataset import Dataset
    from datasets import Datasets

    if len(dataset_list) == 1:
        if dataset_list[0] == "lidc":
            x_train_filenames = ["tr_cubes_64x64x32.npy"]
            x_val_filenames = ["val_cubes_64x64x32.npy"]
            x_test_filenames = ["ts_cubes_64x64x32.npy"]
            files = [x_train_filenames, x_val_filenames, x_test_filenames]
            dataset = Dataset(data_dir=dataset_map[dataset_list[0]], train_val_test=(0.8, 0.2, 0), file_names=files)  # train_val_test is non relevant as is overwritte
        else:
            dataset = Dataset(data_dir=dataset_map[dataset_list[0]], train_val_test=split, file_names=None)
    else:
        assert mode != "" and (mode == "alternate" or mode == "sequential")
        datasets = []
        for idx in range(len(dataset_list)):

            if dataset_list[idx] == "lidc":
                x_train_filenames = ["tr_cubes_64x64x32.npy"]
                x_val_filenames = ["val_cubes_64x64x32.npy"]
                x_test_filenames = ["ts_cubes_64x64x32.npy"]
                files = [x_train_filenames, x_val_filenames, x_test_filenames]
                dataset = Dataset(data_dir=dataset_map[dataset_list[idx]], train_val_test=(0.8, 0.2, 0), file_names=files)  # train_val_test is non relevant as is overwritte
                datasets.append(dataset)
            else:
                datasets.append(Dataset(data_dir=dataset_map[dataset_list[idx]], train_val_test=split, file_names=None))

        dataset = Datasets(datasets=datasets, mode=mode)

    return dataset


if __name__ == "__main__":
    pass
    # build_dataset(["lidc"], split=(0.33, 0.33, 0.34), mode="alternate")

