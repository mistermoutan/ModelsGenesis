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


def replace_config_param_attributes(config_object, kwargs_dict):

    #!!!
    possible_keys = {"optimizer_ss", "scheduler_ss", "lr_ss", "optimizer_sup", "scheduler_sup", "lr_sup", "patience_sup", "patience_sup_terminate", "patience_ss", "patience_ss_terminate", "loss_function_sup", "mode"}
    for key, value in kwargs_dict.items():
        assert isinstance(key, str)
        if key not in possible_keys:
            continue
        if not hasattr(config_object, key):
            raise AttributeError("Config does not have this attribute")
        assert type(value) == type(getattr(config_object, key)), "{}: Trying to replace a {} attribute by a {}".format(key, str(type(getattr(config_object, key))), str(type(value)))
        print("REPLACING {} from {} to {} in Config".format(key, getattr(config_object, key), value))
        setattr(config_object, key, value)


dataset_map = {"lidc": "/work1/s182312/lidc_idri/np_cubes"}  # /work1/s182312/lidc_idri/np_cubes"}


def build_dataset(dataset_list: list, split: tuple):

    from dataset import Dataset

    if len(dataset_list) == 1:
        if dataset_list[0] == "lidc":
            x_train_filenames = ["tr_cubes_64x64x32.npy"]
            x_val_filenames = ["val_cubes_64x64x32.npy"]
            x_test_filenames = ["ts_cubes_64x64x32.npy"]
            files = [x_train_filenames, x_val_filenames, x_test_filenames]
            dataset = Dataset(data_dir=dataset_map[dataset_list[0]], train_val_test=(0.8, 0.2, 0), file_names=files)  # train_val_test is non relevant as is overwritte
        else:
            dataset = Dataset(data_dir=dataset_map[dataset_list[0]], train_val_test=split, file_names=None)

        return dataset

    else:
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

        return datasets


def build_kwargs_dict(args_object, search_for_params=True, **kwargs):
    """[summary]

    Args:
        args_object ([type]): argparse object
        kwargs:
            - get_dataset
            - search_for_split
            - get_directory
    """

    kwargs_dict = {}

    if kwargs.get("get_dataset", False):
        assert len(args_object.dataset) > 0, "Specify dataset(s) with -d, check keys of dataset_map.py"
        kwargs_dict["dataset"] = args_object.dataset
        if len(args_object.dataset) > 1:
            assert args_object.mode is not None, "Specify how sampling should be done with --mode: sequential or alternate"
            assert args_object.mode in ("alternate", "sequential")
            kwargs_dict["mode"] = args_object.mode
        else:
            assert args_object.mode is None, "Don't Specify mode if you are only using 1 dataset"

    if kwargs.get("search_for_split", False):
        if args_object.tr_val_ts_split is not None:
            kwargs_dict["split"] = tuple(args_object.tr_val_ts_split)

    if kwargs.get("get_directory", False):
        assert args_object.directory is not None, "Specify --directory of model weights to load"
        if not os.path.isdir(args_object.directory):
            raise NotADirectoryError("Make sure directory exists")
        kwargs_dict["directory"] = args_object.directory

    #! update possible keys in replace_config_param_attributes if you add params
    if search_for_params:
        if isinstance(args_object.optimizer_ss, str):
            kwargs_dict["optimizer_ss"] = args_object.optimizer_ss
        if isinstance(args_object.scheduler_ss, str):
            kwargs_dict["scheduler_ss"] = args_object.scheduler_ss
        if isinstance(args_object.lr_ss, float):
            kwargs_dict["lr_ss"] = args_object.lr_ss
        if isinstance(args_object.patience_ss, int):
            kwargs_dict["patience_ss"] = args_object.patience_ss
        if isinstance(args_object.patience_ss_terminate, int):
            kwargs_dict["patience_ss_terminate"] = args_object.patience_ss_terminate
        if isinstance(args_object.optimizer_sup, str):
            kwargs_dict["optimizer_sup"] = args_object.optimizer_sup
        if isinstance(args_object.scheduler_sup, str):
            kwargs_dict["scheduler_sup"] = args_object.scheduler_sup
        if isinstance(args_object.lr_sup, float):
            kwargs_dict["lr_sup"] = args_object.lr_sup
        if isinstance(args_object.patience_sup, int):
            kwargs_dict["patience_sup"] = args_object.patience_sup
        if isinstance(args_object.patience_sup_terminate, int):
            kwargs_dict["patience_sup_terminate"] = args_object.patience_sup_terminate
        if isinstance(args_object.loss_function_sup, str):
            kwargs_dict["loss_function_sup"] = args_object.loss_function_sup

    return kwargs_dict


if __name__ == "__main__":
    pass
    # build_dataset(["lidc"], split=(0.33, 0.33, 0.34), mode="alternate")

