import os

from finetune_config import FineTuneConfig
from config import models_genesis_config
from dataset import Dataset
from datasets import Datasets
from finetune import Trainer
from utils import make_dir, save_object, load_object, replace_obj_attributes

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

    config = models_genesis_config()
    # config.optimizer_ss = "adam"

    if kwargs:
        replace_obj_attributes(config, **kwargs)

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

    config = models_genesis_config()
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
    """


def use_provided_weights_and_finetune_on_dataset_without_ss(**kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]

    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()  # alphabetical, IF YOU DO NOT MAINTAIN ORDER A DIFFERENT TASK DIR IS CREATED FOR SAME DATASETS USED: eg: [lidc , brats] vs [brats, lids]
    split = kwargs_dict_.get("split", (0.65, 0.15, 0.1))
    mode = kwargs_dict_.get("mode", "")
    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)

    if len(dataset_list) == 1:
        dataset = Dataset(data_dir=dataset_map[dataset_list[0]], train_val_test=split, file_names=None)
    else:
        assert mode != ""
        datasets = [Dataset(data_dir=dataset_map[dataset_list[idx]], train_val_test=split, file_names=None) for idx in range((dataset_list))]
        dataset = Datasets(datasets=datasets, mode=mode)

    config = FineTuneConfig(data_dir="", task="FROM_PROVIDED_WEIGHTS_SUP_ONLY{}".format(datasets_used_str), self_supervised=False, supervised=True)
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
    split = kwargs_dict_.get("split", (0.65, 0.15, 0.1))
    mode = kwargs_dict_.get("mode", "")
    datasets_used_str = "_" + "_".join(i for i in dataset_list) + "_" + mode if mode != "" else "_" + "_".join(i for i in dataset_list)

    if len(dataset_list) == 1:
        dataset = Dataset(data_dir=dataset_map[dataset_list[0]], train_val_test=split, file_names=None)
    else:
        assert mode != ""
        dataset_list.sort()
        datasets = [Dataset(data_dir=dataset_map[dataset_list[idx]], train_val_test=split, file_names=None) for idx in range((dataset_list))]
        dataset = Datasets(datasets=datasets, mode=mode)

    config = FineTuneConfig(data_dir="", task="FROM_PROVIDED_WEIGHTS_SS_AND_SUP{}".format(datasets_used_str), self_supervised=True, supervised=True)
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


def use_model_weights_and_finetune_on_dataset_without_ss():
    pass


def use_model_weights_and_finetune_on_dataset_with_ss():
    pass


def train_from_scratch_on_dataset():
    pass


if __name__ == "__main__":

    import argparse
    from dataset_map import dataset_map

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--command", required=True, dest="command", type=str)
    parser.add_argument("--run", required=False, dest="run", default=None, type=int)
    parser.add_argument("-d", "--dataset", nargs="+", required=False, dest="dataset", default=[])  # python arg.py -l 1234 2345 3456 4567
    parser.add_argument("-m", "--mode", required=False, dest="mode", default=None, type=str)
    parser.add_argument("--split", nargs="+", required=False, dest="tr_val_ts_split", default=None, type=float)
    parser.add_argument("-opt_ss", "--optimizer_ss", required=False, dest="optimizer_ss", type=str)
    parser.add_argument("-opt_sup", "--optimizer_sup", required=False, dest="optimizer_sup", type=str)
    # model argument?
    args = parser.parse_args()

    # TODO: Add optim and scheduler handling, initial lr
    # TODO: WATCH OUT FOR LIDC WHICH IS ONLY 3 FILES: TR, VAL and TEST

    if args.command == "replicate_model_genesis_pretrain":
        print("STARTING REPLICATION OF RESULTS EXPERIMENT")
        replication_of_results_pretrain()

    elif args.command == "resume_model_genesis_pretrain":
        assert args.run is not None, "You have to specify which --run to resume (int)"
        print("RESUMING REPLICATION OF RESULTS EXPERIMENT FROM RUN {}".format(args.run))
        resume_replication_of_results_pretrain(args.run)

    elif args.command == "finetune_from_provided_weights_no_ss":

        kwargs_dict = {}
        assert len(args.dataset) > 0, "Specify dataset(s) with -d, check keys of dataset_map.py"
        kwargs_dict["dataset"] = args.dataset
        if len(args.dataset > 1):
            assert args.mode is not None, "Specify how sampling should be done with --mode: sequential or alternate"
            kwargs_dict["mode"] = args.mode
        else:
            assert args.mode is None, "Don't Specify mode if you are only using 1 dataset"

        if args.tr_val_ts_split is not None:
            kwargs_dict["split"] = tuple(args.tr_val_ts_split)

        use_provided_weights_and_finetune_on_dataset_without_ss(kwargs_dict=kwargs_dict)

    elif args.command == "resume_finetune_from_provided_weights_no_ss":

        assert args.run is not None, "You have to specify which --run to resume (int)"
        kwargs_dict = {}
        assert len(args.dataset) > 0, "Specify dataset(s) with -d, check keys of dataset_map.py"
        kwargs_dict["dataset"] = args.dataset
        if len(args.dataset > 1):
            assert args.mode is not None, "Specify how sampling should be done with --mode: sequential or alternate"
            kwargs_dict["mode"] = args.mode
        else:
            assert args.mode is None, "Don't Specify mode if you are only using 1 dataset"

        print("RESUMING FINETUNE FROM PROVIDED WEIGHTS EXPERIMENT WITH NO SS FROM RUN {}".format(args.run))
        print("DATASET: {} // MODE: {}".format(kwargs_dict["dataset"], kwargs_dict["mode"]))
        resume_use_provided_weights_and_finetune_on_dataset_without_ss(run_nr=args.run, kwargs_dict=kwargs_dict)

    elif args.command == "finetune_from_provided_weights_with_ss":

        kwargs_dict = {}
        assert len(args.dataset) > 0, "Specify dataset(s) with -d, check keys of dataset_map.py"
        kwargs_dict["dataset"] = args.dataset
        if len(args.dataset > 1):
            assert args.mode is not None, "Specify how sampling should be done with --mode: sequential or alternate"
            kwargs_dict["mode"] = args.mode
        else:
            assert args.mode is None, "Don't Specify mode if you are only using 1 dataset"

        if args.tr_val_ts_split is not None:
            kwargs_dict["split"] = tuple(args.tr_val_ts_split)

        use_provided_weights_and_finetune_on_dataset_with_ss(kwargs_dict=kwargs_dict)

    elif args.command == "resume_finetune_from_provided_weights_with_ss":

        assert args.run is not None, "You have to specify which --run to resume (int)"
        kwargs_dict = {}
        assert len(args.dataset) > 0, "Specify dataset(s) with -d, check keys of dataset_map.py"
        kwargs_dict["dataset"] = args.dataset
        if len(args.dataset > 1):
            assert args.mode is not None, "Specify how sampling should be done with --mode: sequential or alternate"
            kwargs_dict["mode"] = args.mode
        else:
            assert args.mode is None, "Don't Specify mode if you are only using 1 dataset"

        print("RESUMING FINETUNE FROM PROVIDED WEIGHTS EXPERIMENT WITH SS FROM RUN {}".format(args.run))
        print("DATASET: {} // MODE: {}".format(kwargs_dict["dataset"], kwargs_dict["mode"]))
        resume_use_provided_weights_and_finetune_on_dataset_with_ss(run_nr=args.run, kwargs_dict=kwargs_dict)

    elif args.command == "IMPLEMENT THIS":
        pass

    else:
        raise ValueError("Input a valid command")
