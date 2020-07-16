from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

import os
from finetune_config import FineTuneConfig
from config import models_genesis_config
from dataset import Dataset
from datasets import Datasets
from finetune import Trainer
from evaluate import Tester
from utils import *

# script to run experiments

# TODO BUILD CLI:
#       Their Replication:
#           - Replicate model genesis
#           - USe their pretrained weights to test on 5 datasets they provide
#       Me:
#           - Use their framework w/ different dataset (for scratch training)
#           - Use the pretrrained models obtained to test on different datasets
#           - Move into modality , CT and MRI


def replication_of_results_pretrain(**kwargs):

    config = models_genesis_config(False)
    kwargs_dict_ = kwargs["kwargs_dict"]
    replace_config_param_attributes(config, kwargs_dict_)

    config.display()
    save_object(config, "config", config.object_dir)

    x_train_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.train_fold]
    x_val_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.valid_fold]
    x_test_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.test_fold]  # Dont know in what sense they use this for
    files = [x_train_filenames, x_val_filenames, x_test_filenames]
    dataset = Dataset(config.data_dir, train_val_test=(0.8, 0.2, 0), file_names=files)  # train_val_test is non relevant as is overwritten by files

    trainer_mg_replication = Trainer(config, dataset)
    trainer_mg_replication.load_model(from_scratch=True)
    trainer_mg_replication.finetune_self_supervised()
    trainer_mg_replication.add_hparams_to_writer()
    trainer_mg_replication.get_stats()


def resume_replication_of_results_pretrain(run_nr: int, **kwargs):

    config = models_genesis_config(False)
    kwargs_dict_ = kwargs.get("kwargs_dict", False)
    if kwargs_dict_ is not False:
        replace_config_param_attributes(config, kwargs_dict_)
    config.override_dirs(run_nr)  # its key we get object_dir corresponding to the run to fetch the correct config object saved

    # ensure we are not resuming with a different config
    if os.path.isfile(os.path.join(config.object_dir, "config.pkl")):
        config = load_object(os.path.join(config.object_dir, "config.pkl"))  #!
    else:
        # not raising error because some experimetns were done before with saving the object
        print("NO PREVIOUS CONFIG FOUND at {}".format(config.object_dir))

    config.resume_ss = True
    # config.scheduler_ss = "ReduceLROnPlateau"
    config.display()

    # for replication the datasets stay the same
    x_train_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.train_fold]
    x_val_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.valid_fold]
    x_test_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.test_fold]  # Dont know in what sense they use this for
    files = [x_train_filenames, x_val_filenames, x_test_filenames]
    dataset = Dataset(config.data_dir, train_val_test=(0.8, 0.2, 0), file_names=files)  # train_val_test is non relevant as is overwritten by files

    trainer_mg_replication = Trainer(config, dataset)
    trainer_mg_replication.load_model(from_latest_checkpoint=True)  # still requires override dirs to find the specific checkpoint to resume from
    trainer_mg_replication.finetune_self_supervised()
    trainer_mg_replication.add_hparams_to_writer()
    trainer_mg_replication.get_stats()

    """
    --- 
    PRETRAIN MODEL ON DIFFERENT DATASET WITH MG FRAMEWORK
    """


def pretrain_mg_framework_specific_dataset(**kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()  # alphabetical, IF YOU DO NOT MAINTAIN ORDER A DIFFERENT TASK DIR IS CREATED FOR SAME DATASETS USED: eg: [lidc , brats] vs [brats, lids]
    split = kwargs_dict_.get("split", (0.7, 0.1, 0.2))
    mode = kwargs_dict_.get("mode", "")
    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)

    dataset = build_dataset(dataset_list=dataset_list, split=split)
    config = models_genesis_config(True, task="PRETRAIN_MG_FRAMEWORK{}".format(datasets_used_str))
    replace_config_param_attributes(config, kwargs_dict_)
    config.display()

    save_object(config, "config", config.object_dir)
    save_object(dataset, "dataset", config.object_dir)

    trainer_mg_replication = Trainer(config, dataset)
    trainer_mg_replication.load_model(from_scratch=True)
    trainer_mg_replication.finetune_self_supervised()
    trainer_mg_replication.add_hparams_to_writer()
    trainer_mg_replication.get_stats()


def resume_pretrain_mg_framework_specific_dataset(run_nr: int, **kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()  # alphabetical, IF YOU DO NOT MAINTAIN ORDER A DIFFERENT TASK DIR IS CREATED FOR SAME DATASETS USED: eg: [lidc , brats] vs [brats, lids]
    mode = kwargs_dict_.get("mode", "")
    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)

    config = models_genesis_config(True, task="PRETRAIN_MG_FRAMEWORK{}".format(datasets_used_str))
    config.override_dirs(run_nr)

    if os.path.isfile(os.path.join(config.object_dir, "config.pkl")):
        config = load_object(os.path.join(config.object_dir, "config.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find CONFIG object pickle. Did you specify a valid run number?")

    if os.path.isfile(os.path.join(config.object_dir, "dataset.pkl")):
        dataset = load_object(os.path.join(config.object_dir, "dataset.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find DATASET object pickle. Did you specify a valid run number?")

    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_ss = True
    config.display()

    trainer_mg_replication = Trainer(config, dataset)
    trainer_mg_replication.load_model(from_latest_checkpoint=True)
    trainer_mg_replication.finetune_self_supervised()
    trainer_mg_replication.add_hparams_to_writer()
    trainer_mg_replication.get_stats()

    """
    ---
    """


def use_provided_weights_and_finetune_on_dataset_without_ss(**kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]

    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()  # alphabetical, IF YOU DO NOT MAINTAIN ORDER A DIFFERENT TASK DIR IS CREATED FOR SAME DATASETS USED: eg: [lidc , brats] vs [brats, lids]
    split = kwargs_dict_.get("split", (0.7, 0.1, 0.2))
    mode = kwargs_dict_.get("mode", "")
    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)

    dataset = build_dataset(dataset_list=dataset_list, split=split)

    config = FineTuneConfig(data_dir="", task="FROM_PROVIDED_WEIGHTS_SUP_ONLY{}".format(datasets_used_str), self_supervised=False, supervised=True)
    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_from_provided_weights = True  # Redundant, just for logging purposes
    config.display()

    save_object(config, "config", config.object_dir)
    save_object(dataset, "dataset", config.object_dir)

    trainer = Trainer(config, dataset)
    trainer.load_model(from_provided_weights=True)
    trainer.finetune_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()


def resume_use_provided_weights_and_finetune_on_dataset_without_ss(run_nr: int, **kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]

    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()
    mode = kwargs_dict_.get("mode", "")
    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)
    config = FineTuneConfig(data_dir="", task="FROM_PROVIDED_WEIGHTS_SUP_ONLY{}".format(datasets_used_str), self_supervised=False, supervised=True)
    config.override_dirs(run_nr)  # its key we get object_dir corresponding to the run to fetch the correct config object saved

    if os.path.isfile(os.path.join(config.object_dir, "config.pkl")):
        config = load_object(os.path.join(config.object_dir, "config.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find CONFIG object pickle. Did you specify a valid run number?")

    if os.path.isfile(os.path.join(config.object_dir, "dataset.pkl")):
        dataset = load_object(os.path.join(config.object_dir, "dataset.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find DATASET object pickle. Did you specify a valid run number?")

    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_sup = True
    config.display()
    trainer = Trainer(config, dataset)
    trainer.load_model(from_latest_checkpoint=True)
    trainer.finetune_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()

    """
    ---
    """


def use_provided_weights_and_finetune_on_dataset_with_ss(**kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]

    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()
    split = kwargs_dict_.get("split", (0.7, 0.1, 0.2))
    mode = kwargs_dict_.get("mode", "")
    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)

    dataset = build_dataset(dataset_list=dataset_list, split=split)

    config = FineTuneConfig(data_dir="", task="FROM_PROVIDED_WEIGHTS_SS_AND_SUP{}".format(datasets_used_str), self_supervised=True, supervised=True)
    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_from_provided_weights = True  # Redundant, just for logging purposes
    config.display()

    save_object(config, "config", config.object_dir)
    save_object(dataset, "dataset", config.object_dir)

    trainer = Trainer(config, dataset)
    trainer.load_model(from_provided_weights=True)
    trainer.finetune_self_supervised()
    trainer.load_model(from_latest_improvement_ss=True)
    trainer.finetune_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()


def resume_use_provided_weights_and_finetune_on_dataset_with_ss(run_nr: int, **kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]

    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()
    mode = kwargs_dict_.get("mode", "")
    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)
    config = FineTuneConfig(data_dir="", task="FROM_PROVIDED_WEIGHTS_SS_AND_SUP{}".format(datasets_used_str), self_supervised=True, supervised=True)
    config.override_dirs(run_nr)  # its key we get object_dir corresponding to the run to fetch the correct config object saved

    if os.path.isfile(os.path.join(config.object_dir, "config.pkl")):
        config = load_object(os.path.join(config.object_dir, "config.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find CONFIG object pickle. Did you specify a valid run number?")

    if os.path.isfile(os.path.join(config.object_dir, "dataset.pkl")):
        dataset = load_object(os.path.join(config.object_dir, "dataset.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find DATASET object pickle. Did you specify a valid run number?")

    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_ss = True
    config.resume_sup = True
    config.display()

    trainer = Trainer(config, dataset)
    completed_ss = trainer.ss_has_been_completed()

    if not completed_ss:
        trainer.load_model(from_latest_checkpoint=True)
        trainer.finetune_self_supervised()
        trainer.load_model(from_latest_improvement_ss=True)
        trainer.finetune_supervised()
        trainer.add_hparams_to_writer()
        trainer.get_stats()
    else:
        trainer.load_model(from_latest_checkpoint=True)
        trainer.finetune_supervised()
        trainer.add_hparams_to_writer()
        trainer.get_stats()

    """
    ---
    """


def use_model_weights_and_finetune_on_dataset_without_ss(**kwargs):
    # pass it the directory of the task that the model you want to resume from is

    kwargs_dict_ = kwargs["kwargs_dict"]
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()
    model_weights_dir = kwargs_dict["directory"]
    split = kwargs_dict_.get("split", (0.7, 0.1, 0.2))
    mode = kwargs_dict_.get("mode", "")

    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)
    dataset = build_dataset(dataset_list=dataset_list, split=split)

    config = FineTuneConfig(data_dir="", task="FROM_{}_SUP_ONLY{}".format(model_weights_dir, datasets_used_str), self_supervised=False, supervised=True)
    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_from_specific_model = True  # Redundant, just for logging purposes
    config.display()

    save_object(config, "config", config.object_dir)
    save_object(dataset, "dataset", config.object_dir)

    trainer = Trainer(config, dataset)
    trainer.load_model(from_directory=True, directory=model_weights_dir)
    trainer.finetune_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()


def resume_use_model_weights_and_finetune_on_dataset_without_ss(run_nr: int, **kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()
    model_weights_dir = kwargs_dict["directory"]  # to find the task dir to resume from
    mode = kwargs_dict_.get("mode", "")
    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)

    config = FineTuneConfig(data_dir="", task="FROM_{}_SUP_ONLY{}".format(model_weights_dir, datasets_used_str), self_supervised=False, supervised=True)
    config.override_dirs(run_nr)

    if os.path.isfile(os.path.join(config.object_dir, "config.pkl")):
        config = load_object(os.path.join(config.object_dir, "config.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find CONFIG object pickle. Did you specify a valid run number?")

    if os.path.isfile(os.path.join(config.object_dir, "dataset.pkl")):
        dataset = load_object(os.path.join(config.object_dir, "dataset.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find DATASET object pickle. Did you specify a valid run number?")

    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_sup = True
    config.display()

    trainer = Trainer(config, dataset)
    trainer.load_model(from_latest_checkpoint=True)
    trainer.finetune_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()

    """
    ---
    """


def use_model_weights_and_finetune_on_dataset_with_ss(**kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()
    model_weights_dir = kwargs_dict["directory"]
    split = kwargs_dict_.get("split", (0.7, 0.1, 0.2))
    mode = kwargs_dict_.get("mode", "")

    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)
    dataset = build_dataset(dataset_list=dataset_list, split=split)

    config = FineTuneConfig(data_dir="", task="FROM_{}_SS_AND_SUP{}".format(model_weights_dir, datasets_used_str), self_supervised=True, supervised=True)
    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_from_specific_model = True  # Redundant, just for logging purposes
    config.display()

    save_object(config, "config", config.object_dir)
    save_object(dataset, "dataset", config.object_dir)

    trainer = Trainer(config, dataset)
    trainer.load_model(from_directory=True, directory=model_weights_dir)
    trainer.finetune_self_supervised()
    trainer.load_model(from_latest_improvement_ss=True)  # here it's already loading from the dir of the task
    trainer.finetune_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()


def resume_use_model_weights_and_finetune_on_dataset_with_ss(run_nr: int, **kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()
    model_weights_dir = kwargs_dict["directory"]
    mode = kwargs_dict_.get("mode", "")
    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)

    config = FineTuneConfig(data_dir="", task="FROM_{}_SS_AND_SUP{}".format(model_weights_dir, datasets_used_str), self_supervised=True, supervised=True)
    config.override_dirs(run_nr)

    if os.path.isfile(os.path.join(config.object_dir, "config.pkl")):
        config = load_object(os.path.join(config.object_dir, "config.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find CONFIG object pickle. Did you specify a valid run number?")

    if os.path.isfile(os.path.join(config.object_dir, "dataset.pkl")):
        dataset = load_object(os.path.join(config.object_dir, "dataset.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find DATASET object pickle. Did you specify a valid run number?")

    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_ss = True
    config.resume_sup = True
    config.display()

    trainer = Trainer(config, dataset)
    completed_ss = trainer.ss_has_been_completed()

    if not completed_ss:
        trainer.load_model(from_latest_checkpoint=True)
        trainer.finetune_self_supervised()
        trainer.load_model(from_latest_improvement_ss=True)
        trainer.finetune_supervised()
        trainer.add_hparams_to_writer()
        trainer.get_stats()
    else:
        trainer.load_model(from_latest_checkpoint=True)
        trainer.finetune_supervised()
        trainer.add_hparams_to_writer()
        trainer.get_stats()

    """
    ---
    """


def train_from_scratch_on_dataset_no_ss(**kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]

    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()  # alphabetical, IF YOU DO NOT MAINTAIN ORDER A DIFFERENT TASK DIR IS CREATED FOR SAME DATASETS USED: eg: [lidc , brats] vs [brats, lids]
    split = kwargs_dict_.get("split", (0.7, 0.1, 0.2))
    mode = kwargs_dict_.get("mode", "")

    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)

    dataset = build_dataset(dataset_list=dataset_list, split=split)
    config = FineTuneConfig(data_dir="", task="FROM_SCRATCH_NO_SS{}".format(datasets_used_str), self_supervised=False, supervised=True)
    replace_config_param_attributes(config, kwargs_dict_)
    config.from_scratch = True  # Redundant, just for logging purposes
    config.display()

    save_object(config, "config", config.object_dir)
    save_object(dataset, "dataset", config.object_dir)

    trainer = Trainer(config, dataset)
    trainer.load_model(from_scratch=True)
    trainer.finetune_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()


def resume_train_from_scratch_on_dataset_no_ss(run_nr: int, **kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]

    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()  # alphabetical, IF YOU DO NOT MAINTAIN ORDER A DIFFERENT TASK DIR IS CREATED FOR SAME DATASETS USED: eg: [lidc , brats] vs [brats, lids]
    mode = kwargs_dict_.get("mode", "")
    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)
    config = FineTuneConfig(data_dir="", task="FROM_SCRATCH_NO_SS{}".format(datasets_used_str), self_supervised=False, supervised=True)
    config.override_dirs(run_nr)

    if os.path.isfile(os.path.join(config.object_dir, "config.pkl")):
        config = load_object(os.path.join(config.object_dir, "config.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find CONFIG object pickle. Did you specify a valid run number?")

    if os.path.isfile(os.path.join(config.object_dir, "dataset.pkl")):
        dataset = load_object(os.path.join(config.object_dir, "dataset.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find DATASET object pickle. Did you specify a valid run number?")

    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_sup = True
    config.display()

    trainer = Trainer(config, dataset)
    trainer.load_model(from_latest_checkpoint=True)
    trainer.finetune_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()

    """
    ---
    """


def train_from_scratch_on_dataset_with_ss(**kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()  # alphabetical, IF YOU DO NOT MAINTAIN ORDER A DIFFERENT TASK DIR IS CREATED FOR SAME DATASETS USED: eg: [lidc , brats] vs [brats, lids]
    split = kwargs_dict_.get("split", (0.7, 0.1, 0.2))
    mode = kwargs_dict_.get("mode", "")

    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)
    dataset = build_dataset(dataset_list=dataset_list, split=split)
    config = FineTuneConfig(data_dir="", task="FROM_SCRATCH_WITH_SS{}".format(datasets_used_str), self_supervised=True, supervised=True)
    replace_config_param_attributes(config, kwargs_dict_)
    config.from_scratch = True  # Redundant, just for logging purposes
    config.display()

    save_object(config, "config", config.object_dir)
    save_object(dataset, "dataset", config.object_dir)

    trainer = Trainer(config, dataset)
    trainer.load_model(from_scratch=True)
    trainer.finetune_self_supervised()
    trainer.load_model(from_latest_improvement_ss=True)  # here it's already loading from the dir of the task
    trainer.finetune_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()


def resume_train_from_scratch_on_dataset_with_ss(run_nr: int, **kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()  # alphabetical, IF YOU DO NOT MAINTAIN ORDER A DIFFERENT TASK DIR IS CREATED FOR SAME DATASETS USED: eg: [lidc , brats] vs [brats, lids]
    mode = kwargs_dict_.get("mode", "")
    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)
    config = FineTuneConfig(data_dir="", task="FROM_SCRATCH_WITH_SS{}".format(datasets_used_str), self_supervised=True, supervised=True)
    config.override_dirs(run_nr)

    if os.path.isfile(os.path.join(config.object_dir, "config.pkl")):
        config = load_object(os.path.join(config.object_dir, "config.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find CONFIG object pickle. Did you specify a valid run number?")

    if os.path.isfile(os.path.join(config.object_dir, "dataset.pkl")):
        dataset = load_object(os.path.join(config.object_dir, "dataset.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find DATASET object pickle. Did you specify a valid run number?")

    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_sup = True
    config.resume_ss = True
    config.display()

    trainer = Trainer(config, dataset)
    completed_ss = trainer.ss_has_been_completed()

    if not completed_ss:
        trainer.load_model(from_latest_checkpoint=True)
        trainer.finetune_self_supervised()
        trainer.load_model(from_latest_improvement_ss=True)
        trainer.finetune_supervised()
        trainer.add_hparams_to_writer()
        trainer.get_stats()
    else:
        trainer.load_model(from_latest_checkpoint=True)
        trainer.finetune_supervised()
        trainer.add_hparams_to_writer()
        trainer.get_stats()

    def test(**kwargs):

        kwargs_dict_ = kwargs["kwargs_dict"]
        task_dirs = get_task_dirs()
        test_config = TestConfig()

        for task_dir in task_dirs:
            config_object = get_config_object_of_task_dir(task_dir)
            dataset_object = get_dataset_object_of_task_dir(task_dir)
            tester = Tester(config, dataset)
            tester.test_segmentation()

        # if not files:
        #    continue
        # else:
        #    print("ROOT ", root)
        #    print("DIRS", dirs)
        #    print(files)

        tester = Tester()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--command", required=True, dest="command", type=str)
    parser.add_argument("--run", required=False, dest="run", default=None, type=int)
    parser.add_argument("-d", "--dataset", nargs="+", required=False, dest="dataset", default=[])  # python arg.py -l 1234 2345 3456 4567
    parser.add_argument("--mode", required=False, dest="mode", default=None, type=str)
    parser.add_argument("--directory", required=False, dest="directory", type=str, default=None)
    parser.add_argument("--split", nargs="+", required=False, dest="tr_val_ts_split", default=None, type=str)

    parser.add_argument("-opt_ss", "--optimizer_ss", required=False, dest="optimizer_ss", type=str)
    parser.add_argument("-sch_ss", "--scheduler_ss", required=False, dest="scheduler_ss", type=str)
    parser.add_argument("-lr_ss", "--learning_rate_ss", required=False, dest="lr_ss", type=float)
    parser.add_argument("--patience_ss_terminate", required=False, dest="patience_ss_terminate", type=int)
    parser.add_argument("--patience_ss", required=False, dest="patience_ss", type=int)
    parser.add_argument("-opt_sup", "--optimizer_sup", required=False, dest="optimizer_sup", type=str)
    parser.add_argument("-sch_sup", "--scheduler_sup", required=False, dest="scheduler_sup", type=str)
    parser.add_argument("-lr_sup", "--learning_rate_sup", required=False, dest="lr_sup", type=float)
    parser.add_argument("--patience_sup_terminate", required=False, dest="patience_sup_terminate", type=int)
    parser.add_argument("--patience_sup", required=False, dest="patience_sup", type=int)
    parser.add_argument("--loss_function_sup", required=False, dest="loss_function_sup", type=str)
    # model argument?
    parser.add_argument("--task_name", required=False, dest="task_name", type=str, default=None)

    args = parser.parse_args()

    # TODO: Add optim and scheduler handling, initial lr, model change

    if args.command == "replicate_model_genesis_pretrain":
        print("STARTING REPLICATION OF RESULTS EXPERIMENT")
        kwargs_dict = build_kwargs_dict(args)
        replication_of_results_pretrain(kwargs_dict=kwargs_dict)

    elif args.command == "resume_model_genesis_pretrain":
        assert args.run is not None, "You have to specify which --run to resume (int)"
        kwargs_dict = build_kwargs_dict(args)
        print("RESUMING REPLICATION OF RESULTS EXPERIMENT FROM RUN {}".format(args.run))
        resume_replication_of_results_pretrain(args.run, kwargs_dict=kwargs_dict)

    elif args.command == "finetune_from_provided_weights_no_ss":

        kwargs_dict = build_kwargs_dict(args, get_dataset=True, search_for_split=True)
        use_provided_weights_and_finetune_on_dataset_without_ss(kwargs_dict=kwargs_dict)

    elif args.command == "resume_finetune_from_provided_weights_no_ss":

        assert args.run is not None, "You have to specify which --run to resume (int)"
        kwargs_dict = build_kwargs_dict(args, get_dataset=True)
        print("RESUMING FINETUNE FROM PROVIDED WEIGHTS EXPERIMENT WITH NO SS FROM RUN {}".format(args.run))
        print("DATASET: {} // MODE: {}".format(kwargs_dict["dataset"], args.mode))
        resume_use_provided_weights_and_finetune_on_dataset_without_ss(run_nr=args.run, kwargs_dict=kwargs_dict)

    elif args.command == "finetune_from_provided_weights_with_ss":

        kwargs_dict = build_kwargs_dict(args, get_dataset=True, search_for_split=True)
        use_provided_weights_and_finetune_on_dataset_with_ss(kwargs_dict=kwargs_dict)

    elif args.command == "resume_finetune_from_provided_weights_with_ss":

        assert args.run is not None, "You have to specify which --run to resume (int)"
        kwargs_dict = build_kwargs_dict(args, get_dataset=True)
        print("RESUMING FINETUNE FROM PROVIDED WEIGHTS EXPERIMENT WITH SS FROM RUN {}".format(args.run))
        print("DATASET: {} // MODE: {}".format(kwargs_dict["dataset"], args.mode))
        resume_use_provided_weights_and_finetune_on_dataset_with_ss(run_nr=args.run, kwargs_dict=kwargs_dict)

        """
        ---
        """

    elif args.command == "pretrain_mg_framework":

        kwargs_dict = build_kwargs_dict(args, get_dataset=True, search_for_split=True)
        pretrain_mg_framework_specific_dataset(kwargs_dict=kwargs_dict)

    elif args.command == "resume_pretrain_mg_framework":

        assert args.run is not None, "You have to specify which --run to resume (int)"
        kwargs_dict = build_kwargs_dict(args, get_dataset=True)
        print("RESUMING PRETRAIN ACCORDING TO MG FRAMEWORK FROM RUN {}".format(args.run))
        print("DATASET: {} // MODE: {}".format(kwargs_dict["dataset"], args.mode))
        resume_pretrain_mg_framework_specific_dataset(run_nr=args.run, kwargs_dict=kwargs_dict)

        """
        ---
        """

    elif args.command == "finetune_from_model_no_ss":

        kwargs_dict = build_kwargs_dict(args, get_dataset=True, search_for_split=True, get_directory=True)
        use_model_weights_and_finetune_on_dataset_without_ss(kwargs_dict=kwargs_dict)

    elif args.command == "resume_finetune_from_model_no_ss":

        assert args.run is not None, "You have to specify which --run to resume (int)"
        kwargs_dict = build_kwargs_dict(args, get_dataset=True, get_directory=True)
        print("RESUMING FINETUNE FROM {} WEIGHTS NO SS FROM RUN {}".format(args.directory, args.run))
        print("DATASET: {} // MODE: {}".format(kwargs_dict["dataset"], args.mode))
        resume_use_model_weights_and_finetune_on_dataset_without_ss(run_nr=args.run, kwargs_dict=kwargs_dict)

        """
        ---
        """

    elif args.command == "finetune_from_model_with_ss":

        kwargs_dict = build_kwargs_dict(args, get_dataset=True, search_for_split=True, get_directory=True)
        use_model_weights_and_finetune_on_dataset_with_ss(kwargs_dict=kwargs_dict)

    elif args.command == "resume_finetune_from_model_with_ss":

        assert args.run is not None, "You have to specify which --run to resume (int)"
        kwargs_dict = build_kwargs_dict(args, get_dataset=True, get_directory=True)
        resume_use_model_weights_and_finetune_on_dataset_with_ss(args.run, kwargs_dict=kwargs_dict)

        """
        ---
        """

    elif args.command == "from_scratch_supervised":

        kwargs_dict = build_kwargs_dict(args, get_dataset=True, search_for_split=True)
        train_from_scratch_on_dataset_no_ss(kwargs_dict=kwargs_dict)

    elif args.command == "resume_from_scratch_supervised":

        assert args.run is not None, "You have to specify which --run to resume (int)"
        kwargs_dict = build_kwargs_dict(args, get_dataset=True)
        resume_train_from_scratch_on_dataset_no_ss(run_nr=args.run, kwargs_dict=kwargs_dict)

    elif args.command == "from_scratch_ss_and_sup":

        kwargs_dict = build_kwargs_dict(args, get_dataset=True, search_for_split=True)
        train_from_scratch_on_dataset_with_ss(kwargs_dict=kwargs_dict)

    elif args.command == "resume_from_scratch_ss_and_sup":

        assert args.run is not None, "You have to specify which --run to resume (int)"
        kwargs_dict = build_kwargs_dict(args, get_dataset=True)
        resume_train_from_scratch_on_dataset_with_ss(run_nr=args.run, kwargs_dict=kwargs_dict)

    elif args.command == "test":
        kwargs_dict = build_kwargs_dict(test=True, search_for_params=False)
        test(kwargs_dict=kwargs_dict)

    else:
        raise ValueError("Input a valid command")
