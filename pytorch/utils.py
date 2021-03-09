import os
import dill
from copy import deepcopy
import torch.nn.functional as F
import pandas as pd


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


def get_config_object_of_task_dir(task_dir):

    if os.path.isfile(os.path.join("objects/", task_dir, "config.pkl")):
        o = load_object(os.path.join("objects/", task_dir, "config.pkl"))
    else:
        o = None
    return o


def get_dataset_object_of_task_dir(task_dir):

    if os.path.isfile(os.path.join("objects/", task_dir, "dataset.pkl")):
        o = load_object(os.path.join("objects/", task_dir, "dataset.pkl"))
    else:
        o = None
    return o


def save_cross_validator_object_of_task_dir(cv, task_dir):

    tmp = task_dir.split("_")
    tmp[-1] = "1/"
    task_dir_run_1 = "_".join(i for i in tmp)
    save_object(cv, "cross_validator", os.path.join("objects/", task_dir_run_1))


def get_cross_validator_object_of_task_dir(task_dir):

    tmp = task_dir.split("_")
    tmp[-1] = "1/"
    task_dir_run_1 = "_".join(i for i in tmp)
    # print(task_dir_run_1)

    if os.path.isfile(os.path.join("objects/", task_dir_run_1, "cross_validator.pkl")):
        o = load_object(os.path.join("objects/", task_dir_run_1, "cross_validator.pkl"))
    else:
        o = None
    return o


def task_dir_already_has_metric_dict_computed(task_dir):

    if os.path.isfile(os.path.join("test_results/", task_dir, "test_results_new0.json")):
        return True
    else:
        return False


def replace_config_param_attributes(config_object, kwargs_dict, ilegal=None):

    #!!!
    possible_keys = {
        "optimizer_ss",
        "scheduler_ss",
        "lr_ss",
        "optimizer_sup",
        "scheduler_sup",
        "lr_sup",
        "patience_sup",
        "patience_sup_terminate",
        "patience_ss",
        "patience_ss_terminate",
        "loss_function_sup",
        "mode",
        "batch_size_ss",
        "batch_size_sup",
        "model",
    }
    for key, value in kwargs_dict.items():
        assert isinstance(key, str)
        if key not in possible_keys:
            continue
        if ilegal is not None:
            if key in ilegal:
                continue
        if not hasattr(config_object, key):
            raise AttributeError("Config does not have this attribute")
        assert type(value) == type(getattr(config_object, key)), "{}: Trying to replace a {} attribute by a {}".format(
            key, str(type(getattr(config_object, key))), str(type(value))
        )
        print("REPLACING {} from {} to {} in Config".format(key, getattr(config_object, key), value))
        setattr(config_object, key, value)


# keys correspondto what dataset should be named on CLI
dataset_map = {
    "lidc": "/work1/s182312/lidc_idri/np_cubes",
    "lidc_80_80_padded": "/work2/s182312/lidc_mini_cubes_for_acs_replication_padded",
    "lidc_acs_provided": "/work2/s182312/lidc_acs_provided",
    "lidc_mg_provided": "/work1/s182312/luna16_extracted_cubes/scale_32",
    "task01_ss": "/work1/s182312/medical_decathlon/Task01_BrainTumour/imagesTr/extracted_cubes_64_64_32_ss",
    "task01_sup": "/work1/s182312/medical_decathlon/Task01_BrainTumour/imagesTr/extracted_cubes_64_64_32_sup",
    "task02_ss": "/work1/s182312/medical_decathlon/Task02_Heart/imagesTr/extracted_cubes_64_64_32_ss",
    "task02_sup": "/work1/s182312/medical_decathlon/Task02_Heart/imagesTr/extracted_cubes_64_64_32_sup",
    "task03_ss": "/work1/s182312/medical_decathlon/Task03_Liver/imagesTr/extracted_cubes_128_128_64_ss",
    "task03_sup": "/work1/s182312/medical_decathlon/Task03_Liver/imagesTr/extracted_cubes_128_128_64_sup",
    "task04_ss": "/work1/s182312/medical_decathlon/Task04_Hippocampus/imagesTr/extracted_cubes_32_32_32_ss",
    "task04_sup": "/work1/s182312/medical_decathlon/Task04_Hippocampus/imagesTr/extracted_cubes_32_32_32_sup",
    "task05_ss": "/work1/s182312/medical_decathlon/Task05_Prostate/imagesTr/extracted_cubes_64_64_16_ss",
    "task05_sup": "/work1/s182312/medical_decathlon/Task05_Prostate/imagesTr/extracted_cubes_64_64_16_sup",
    "task06_ss": "/work1/s182312/medical_decathlon/Task06_Lung/imagesTr/extracted_cubes_64_64_32_ss",
    "task06_sup": "/work1/s182312/medical_decathlon/Task06_Lung/imagesTr/extracted_cubes_64_64_32_sup",
    "task07_ss": "/work1/s182312/medical_decathlon/Task07_Pancreas/imagesTr/extracted_cubes_64_64_32_ss",
    "task07_sup": "/work1/s182312/medical_decathlon/Task07_Pancreas/imagesTr/extracted_cubes_64_64_32_sup",
    "task08_ss": "/work1/s182312/medical_decathlon/Task08_HepaticVessel/imagesTr/extracted_cubes_64_64_32_ss",
    "task08_sup": "/work1/s182312/medical_decathlon/Task08_HepaticVessel/imagesTr/extracted_cubes_64_64_32_sup",
    "task09_ss": "/work1/s182312/medical_decathlon/Task09_Spleen/imagesTr/extracted_cubes_64_64_32_ss",
    "task09_sup": "/work1/s182312/medical_decathlon/Task09_Spleen/imagesTr/extracted_cubes_64_64_32_sup",
    "task10_ss": "/work1/s182312/medical_decathlon/Task10_Colon/imagesTr/extracted_cubes_64_64_32_ss",
    "task10_sup": "/work1/s182312/medical_decathlon/Task10_Colon/imagesTr/extracted_cubes_64_64_32_sup",
    "cellari_heart_sup": "/work1/s182312/heart_mri/datasets/x_cubes_full/extracted_cubes_64_64_12_sup",
    "cellari_heart_sup_10_192": "/work1/s182312/heart_mri/datasets/x_cubes_full/extracted_cubes_10_192_192_12_sup",
    "fake_small": "/home/moutan/Programming/thesis/ModelGenesis_Fork/ModelsGenesis/pytorch/datasets/fake_small",
}

""" dataset_map = {
    "task01_ss": "pytorch/datasets/task01_brats/Task01_BrainTumour/imagesTr/extracted_cubes_64_64_32_ss",
    "task01_sup": "pytorch/datasets/task01_brats/Task01_BrainTumour/imagesTr/extracted_cubes_64_64_32_sup",
    "cellari_heart": "pytorch/datasets/heart_mri/datasets/x_cubes_full/extracted_cubes_64_64_12_sup",
} """


""" dataset_full_cubes_map = {"cellari_heart": "pytorch/datasets/heart_mri/datasets/x_cubes_full"}

dataset_full_cubes_labels_map = {"cellari_heart": "pytorch/datasets/heart_mri/datasets/y_cubes_full"} """
# for tester module and full cube segmentation module
dataset_full_cubes_map = {
    "lidc": "/work2/s182312/lidc_full_cubes/full_cubes",
    "lidc_80_80_padded": "/work2/s182312/lidc_full_cubes/full_cubes",
    "task01_sup": "/work1/s182312/medical_decathlon/Task01_BrainTumour/imagesTr",
    "task02_sup": "/work1/s182312/medical_decathlon/Task02_Heart/imagesTr",
    "task03_sup": "/work1/s182312/medical_decathlon/Task03_Liver/imagesTr",
    "task04_sup": "/work1/s182312/medical_decathlon/Task04_Hippocampus/imagesTr",
    "task05_sup": "/work1/s182312/medical_decathlon/Task05_Prostate/imagesTr",
    "task06_sup": "/work1/s182312/medical_decathlon/Task06_Lung/imagesTr",
    "task07_sup": "/work1/s182312/medical_decathlon/Task07_Pancreas/imagesTr",
    "task08_sup": "/work1/s182312/medical_decathlon/Task08_HepaticVessel/imagesTr",
    "task09_sup": "/work1/s182312/medical_decathlon/Task09_Spleen/imagesTr",
    "task10_sup": "/work1/s182312/medical_decathlon/Task10_Colon/imagesTr",
    "cellari_heart_sup": "/work1/s182312/heart_mri/datasets/x_cubes_full",
    "cellari_heart_sup_10_192": "/work1/s182312/heart_mri/datasets/x_cubes_full",
}

dataset_full_cubes_labels_map = {
    "lidc": "/work2/s182312/lidc_full_cubes/full_cubes_labels",
    "lidc_80_80_padded": "/work2/s182312/lidc_full_cubes/full_cubes_labels",
    "task01_sup": "/work1/s182312/medical_decathlon/Task01_BrainTumour/labelsTr",
    "task02_sup": "/work1/s182312/medical_decathlon/Task02_Heart/labelsTr",
    "task03_sup": "/work1/s182312/medical_decathlon/Task03_Liver/labelsTr",
    "task04_sup": "/work1/s182312/medical_decathlon/Task04_Hippocampus/labelsTr",
    "task05_sup": "/work1/s182312/medical_decathlon/Task05_Prostate/labelsTr",
    "task06_sup": "/work1/s182312/medical_decathlon/Task06_Lung/labelsTr",
    "task07_sup": "/work1/s182312/medical_decathlon/Task07_Pancreas/labelsTr",
    "task08_sup": "/work1/s182312/medical_decathlon/Task08_HepaticVessel/labelsTr",
    "task09_sup": "/work1/s182312/medical_decathlon/Task09_Spleen/labelsTr",
    "task10_sup": "/work1/s182312/medical_decathlon/Task10_Colon/labelsTr",
    "cellari_heart_sup": "/work1/s182312/heart_mri/datasets/y_cubes_full",
    "cellari_heart_sup_10_192": "/work1/s182312/heart_mri/datasets/y_cubes_full",
}


# "task_02": "pytorch/datasets/task02/extracted_cubes",
# "luna": "pytorch/datasets/luna16_cubes",
# /work1/s182312/lidc_idri/np_cubes"}
# pytorch/datasets/lidc_idri_cubes


def get_files_for_train_and_test_for_provided_acs_lidc():

    data_path = "/work2/s182312/lidc_acs_provided"
    names_train = (
        pd.read_csv(os.path.join(data_path, "train_test_split.csv"))["train"]
        .dropna()
        .map(lambda x: os.path.join(data_path, "nodule", x))
        .values
    )
    names_test = (
        pd.read_csv(os.path.join(data_path, "train_test_split.csv"))["test"]
        .dropna()
        .map(lambda x: os.path.join(data_path, "nodule", x))
        .values
    )

    names_train_ = [j.split("/")[-1] for j in names_train]
    names_test_ = [j.split("/")[-1] for j in names_test]

    return names_train_, names_test_


def get_unused_datasets(dataset):

    all_datasets = set(dataset_map.values())
    # print("ALL DATASETS ", all_datasets)
    if isinstance(dataset, list):
        for d in dataset:
            if d.x_data_dir[:-3] in all_datasets:
                all_datasets.remove(d.x_data_dir[:-3])
    else:
        if dataset.x_data_dir[:-3] in all_datasets:
            all_datasets.remove(dataset.x_data_dir[:-3])
    # print("UNUSED DATASETS ", all_datasets)
    return all_datasets


def build_dataset(dataset_list: list, split: tuple, two_dimensional_data=False, use_supervision_transforms=False, data_limit_2d=None):
    """[summary]

    Returns:
        Dataset or [Dataset, Dataset]:
    """

    from dataset import Dataset
    from dataset_2d import Dataset2D

    #  dataset come as [] from CLI
    if len(dataset_list) == 1:
        if dataset_list[0] == "lidc":
            x_train_filenames = ["tr_cubes_64x64x32.npy"]
            x_val_filenames = ["val_cubes_64x64x32.npy"]
            x_test_filenames = ["ts_cubes_64x64x32.npy"]
            files = [x_train_filenames, x_val_filenames, x_test_filenames]
            dataset = Dataset(
                data_dir=dataset_map[dataset_list[0]], train_val_test=(0, 0, 1), file_names=files
            )  # train_val_test is non relevant as is overwritte

        elif dataset_list[0] == "lidc_acs_provided":
            x_train_filenames, x_val_filenames = get_files_for_train_and_test_for_provided_acs_lidc()
            files = [x_train_filenames, x_val_filenames, []]
            dataset = Dataset(data_dir=dataset_map[dataset_list[0]], train_val_test=(0, 0, 1), file_names=files)

        elif dataset_list[0] == "lidc_mg_provided":
            x_train_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in range(5)]
            x_val_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in range(5, 10)]
            files = [x_train_filenames, x_val_filenames, []]
            dataset = Dataset(data_dir=dataset_map[dataset_list[0]], train_val_test=(0, 0, 1), file_names=files)

        else:
            dataset = Dataset(data_dir=dataset_map[dataset_list[0]], train_val_test=split, file_names=None)

        if two_dimensional_data is True:
            dataset = Dataset2D(dataset, limit_of_samples=data_limit_2d)

        if use_supervision_transforms is True:
            dataset.use_supervision_transforms = True

        return dataset

    else:
        datasets = []
        for idx in range(len(dataset_list)):
            if dataset_list[idx] == "lidc":
                x_train_filenames = ["tr_cubes_64x64x32.npy"]
                x_val_filenames = ["val_cubes_64x64x32.npy"]
                x_test_filenames = ["ts_cubes_64x64x32.npy"]
                files = [x_train_filenames, x_val_filenames, x_test_filenames]
                dataset = Dataset(
                    data_dir=dataset_map[dataset_list[idx]], train_val_test=(0, 0, 1), file_names=files
                )  # train_val_test is non relevant as is overwritte
                if two_dimensional_data:
                    dataset = Dataset2D(dataset)
                datasets.append(dataset)
            else:
                dataset = Dataset(data_dir=dataset_map[dataset_list[idx]], train_val_test=split, file_names=None)
                if two_dimensional_data:
                    dataset = Dataset2D(dataset, limit_of_samples=data_limit_2d)
                datasets.append(dataset)

        if use_supervision_transforms is True:
            for idx, dataset in enumerate(datasets):
                dataset.use_supervision_transforms = True
                datasets[idx] = dataset

        return datasets


def get_datasets_used_str(dataset_list, mode, two_dim_data, convert_to_acs=False):

    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)
    if two_dim_data is True:
        datasets_used_str += "_2D"
    if convert_to_acs is True:
        datasets_used_str += "/ACS_Conversion/"
    return datasets_used_str


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

    if args_object.num_cv_folds is not None:
        assert args_object.dataset != "lidc", "This lidc is to be used with the splits as the Model Genesis paper describes"
        kwargs_dict["num_cv_folds"] = args_object.num_cv_folds

    if kwargs.get("get_directory", False):
        assert args_object.directory is not None, "Specify --directory of model weights to load"
        if not os.path.isdir(args_object.directory):
            raise NotADirectoryError("Make sure directory exists")
        kwargs_dict["directory"] = args_object.directory

    if kwargs.get("test", False):
        pass
        # not in use
        # if isinstance(args_object.task_name, str):
        #    kwargs_dict["task_name"] = args_object.directory
        # if isinstance(args_object.directory, str):
        #    kwargs_dict["directory"] = args_object.directory
        # if args_object.dataset != []:
        #    kwargs_dict["dataset"] = args_object.dataset

    # model
    assert args_object.model.lower() in (
        "vnet_mg",
        "unet_2d",
        "unet_acs",
        "unet_3d",
        "fcn_resnet18_acs",
        "fcn_resnet18_acs_pretrained_imgnet",
        "unet_acs_with_cls",
        "unet_acs_axis_aware_decoder",
        "unet_acs_cls_only",
        "unet_acs_cls_only_frozen_encoder",
    )
    kwargs_dict["model"] = args_object.model
    if args_object.model.lower() == "unet_2d":
        if args_object.convert_to_acs is not True:
            assert args_object.two_dimensional_data is True, "Need to work with 2d Data for Unet_2d, pass --two_dimensional_data argument"

    kwargs_dict["two_dimensional_data"] = args_object.two_dimensional_data
    if args_object.convert_to_acs is True:
        assert args_object.two_dimensional_data is False, "You are going to use 3D Data now"

    kwargs_dict["convert_to_acs"] = args_object.convert_to_acs
    kwargs_dict["new_folder"] = args_object.new_folder
    kwargs_dict["use_supervision_transforms"] = args_object.use_supervision_transforms
    kwargs_dict["task_name"] = args_object.task_name
    kwargs_dict["make_acs_kernel_split_adaptive_to_input_dimensions"] = args_object.make_acs_kernel_split_adaptive_to_input_dimensions
    kwargs_dict["data_limit_2d"] = args_object.data_limit_2d
    kwargs_dict["enforce_test_again"] = args_object.enforce_test_again
    kwargs_dict["mini_only"] = args_object.mini_only
    kwargs_dict["full_only"] = args_object.full_only
    kwargs_dict["pool_features"] = args_object.pool_features
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
        if isinstance(args_object.batch_size_ss, int):
            kwargs_dict["batch_size_ss"] = args_object.batch_size_ss
        if isinstance(args_object.batch_size_sup, int):
            kwargs_dict["batch_size_sup"] = args_object.batch_size_sup

    return kwargs_dict


def get_task_dirs():

    task_dirs = []
    for root, dirs, files in os.walk("pretrained_weights"):
        if not files:
            continue
        else:
            root_split = root.split("/")
            task_dir = "/".join(i for i in root_split[1:])
            if task_dir == "":
                print("SHOULD BE PRINTING THE PROVIDED WEIGHTS")
                print("\n", "ROOT", root, "\n", "DIRS ", dirs, "\n", "FILES", files, "\n")
                continue
            task_dirs.append(task_dir)
    return task_dirs


def pad_if_necessary(x, y, min_size=16):

    pad = []
    for idx, i in enumerate(x.shape):
        if idx in (0, 1):
            # not padding batch and channel dims
            pad.insert(0, 0)
            pad.insert(0, 0)
            continue

        if i < min_size:
            resto = min_size % i
        else:
            resto = i % min_size
            # we need do pad to the next multiple of min_size
            if resto != 0:
                base = min_size
                tmp = base
                multiplier = 2
                while tmp < i:
                    tmp = base * multiplier
                    multiplier += 1
                base = tmp
                resto = base - i

        if resto != 0:
            if resto % 2 == 0:
                pad.insert(0, int(resto / 2))
                pad.insert(0, int(resto / 2))
            else:
                maior = int((resto - 1) / 2)
                menor = int(resto - maior)
                pad.insert(0, maior)
                pad.insert(0, menor)
        else:
            pad.insert(0, 0)
            pad.insert(0, 0)

    if set(pad) == {0}:
        # no padding necessary
        return x, y
    pad_tuple = tuple(pad)
    # print(x.shape, y.shape)

    x = F.pad(x, pad_tuple, "constant", 0)
    y = F.pad(y, pad_tuple, "constant", 0)
    # print(x.shape, y.shape)
    return x, y


def pad_if_necessary_one_array(x, min_size=16, return_pad_tuple=False):
    """
    get each dim to min size or multiple of min size
    """
    pad = []
    for idx, i in enumerate(x.shape):
        if idx in (0, 1):
            # not padding batch and channel dims
            pad.insert(0, 0)
            pad.insert(0, 0)
            continue

        if i < min_size:
            resto = min_size % i
        else:
            resto = i % min_size
            # we need do pad to the next multiple of min_size
            if resto != 0:
                base = min_size
                tmp = base
                multiplier = 2
                while tmp < i:
                    tmp = base * multiplier
                    multiplier += 1
                base = tmp
                resto = base - i

        if resto != 0:
            if resto % 2 == 0:
                pad.insert(0, int(resto / 2))
                pad.insert(0, int(resto / 2))
            else:
                maior = int((resto - 1) / 2)
                menor = int(resto - maior)
                pad.insert(0, maior)
                pad.insert(0, menor)
        else:
            pad.insert(0, 0)
            pad.insert(0, 0)

    if set(pad) == {0}:
        # no padding necessary
        if return_pad_tuple is False:
            return x
        else:
            return x, tuple(pad)

    pad_tuple = tuple(pad)

    x = F.pad(x, pad_tuple, "constant", 0)
    if return_pad_tuple is False:
        return x
    else:
        return x, pad_tuple


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert isinstance(size, torch.Size)
    return " × ".join(map(str, size))


def dump_tensors(gpu_only=False):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc

    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print(
                        "%s:%s%s %s"
                        % (type(obj).__name__, " GPU" if obj.is_cuda else "", " pinned" if obj.is_pinned else "", pretty_size(obj.size()))
                    )
                    total_size += obj.numel()
                    del obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print(
                        "%s → %s:%s%s%s%s %s"
                        % (
                            type(obj).__name__,
                            type(obj.data).__name__,
                            " GPU" if obj.is_cuda else "",
                            " pinned" if obj.data.is_pinned else "",
                            " grad" if obj.requires_grad else "",
                            " volatile" if obj.volatile else "",
                            pretty_size(obj.data.size()),
                        )
                    )
                    total_size += obj.data.numel()
                    del obj
        except Exception as e:
            pass
    # print("Total size:", total_size)


if __name__ == "__main__":
    import torch

    a = torch.zeros((1, 1, 130, 128, 12))
    a = pad_if_necessary_one_array(a, return_pad_tuple=False)
    print(a.shape)
    pass
    # build_dataset(["lidc"], split=(0.33, 0.33, 0.34), mode="alternate")
