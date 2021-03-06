from warnings import simplefilter
import torch

simplefilter(action="ignore", category=FutureWarning)

import os
from finetune_config import FineTuneConfig
from config import models_genesis_config
from dataset import Dataset
from finetune import Trainer
from evaluate import Tester
from cross_validator import CrossValidator
from feature_extractor import FeatureExtractor
from utils import *


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
    dataset = Dataset(
        config.data_dir, train_val_test=(0.8, 0.2, 0), file_names=files
    )  # train_val_test is non relevant as is overwritten by files

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
    dataset = Dataset(
        config.data_dir, train_val_test=(0.8, 0.2, 0), file_names=files
    )  # train_val_test is non relevant as is overwritten by files

    trainer_mg_replication = Trainer(config, dataset)
    trainer_mg_replication.load_model(
        from_latest_checkpoint=True
    )  # still requires override dirs to find the specific checkpoint to resume from
    trainer_mg_replication.finetune_self_supervised()
    trainer_mg_replication.add_hparams_to_writer()
    trainer_mg_replication.get_stats()


def replicate_acs_results_fcnresnet18_their_cubes(**kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]
    dataset_list = ["lidc_acs_provided"]
    split = (0.8, 0.2, 0)

    dataset = build_dataset(dataset_list=dataset_list, split=split, two_dimensional_data=False)
    dataset.use_acs_paper_transforms = True  # !!

    config = FineTuneConfig(
        data_dir="",
        task="REPLICATE_ACS_PAPER_THEIR_EXACT_DATA",
        self_supervised=False,
        supervised=True,
        model=kwargs_dict_["model"],
    )

    config.batch_size_sup = 8
    config.nb_epoch_sup = 100
    config.lr_sup = 0.001
    config.milestones = [0.5 * config.nb_epoch_sup, 0.75 * config.nb_epoch_sup]
    config.gamma = 0.1
    config.scheduler_sup = "MultiStepLR"

    # let it run since > than nb_epochs
    config.patience_sup_terminate = 120
    config.from_scratch = True

    config.display()

    save_object(config, "config", config.object_dir)
    save_object(dataset, "dataset", config.object_dir)

    trainer_mg_replication = Trainer(config, dataset)
    trainer_mg_replication.load_model(from_scratch=True)
    trainer_mg_replication.finetune_supervised()
    trainer_mg_replication.add_hparams_to_writer()
    trainer_mg_replication.get_stats()


def replicate_acs_results_fcnresnet18_my_cubes(**kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]
    dataset_list = ["lidc_80_80_padded"]
    split = (0.8, 0.2, 0)
    num_cv_folds = kwargs_dict_.get("num_cv_folds", None)

    dataset = build_dataset(dataset_list=dataset_list, split=split, two_dimensional_data=False)
    dataset.use_acs_paper_transforms = True  # !!

    config = FineTuneConfig(
        data_dir="",
        task="REPLICATE_ACS_PAPER",
        self_supervised=False,
        supervised=True,
        model=kwargs_dict_["model"],
    )
    config.batch_size_sup = 8
    config.nb_epoch_sup = 100
    config.lr_sup = 0.001
    config.milestones = [0.5 * config.nb_epoch_sup, 0.75 * config.nb_epoch_sup]
    config.gamma = 0.1
    config.scheduler_sup = "MultiStepLR"

    # let it run since > than nb_epochs
    config.patience_sup_terminate = 120

    config.from_scratch = True

    if num_cv_folds is not None:
        cv = get_cross_validator_object_of_task_dir(config.task_dir)
        if cv is None:
            if config.experiment_nr == 1:
                cv = CrossValidator(config, dataset, nr_splits=num_cv_folds)
                cv.override_dataset_files_with_splits()
                save_object(cv, "cross_validator", config.object_dir)
                print("RUN 1: Building cross validator")
            else:
                print("TOO LATE TO BRING CROSS VALIDATON IN")

        else:
            cv.set_dataset(dataset)
            cv.override_dataset_files_with_splits()
            # to "loose" used splits as they're popped and needs to be saved in run1 objects
            save_cross_validator_object_of_task_dir(cv, config.task_dir)

    config.display()

    save_object(config, "config", config.object_dir)
    save_object(dataset, "dataset", config.object_dir)

    trainer_mg_replication = Trainer(config, dataset)
    trainer_mg_replication.load_model(from_scratch=True)
    trainer_mg_replication.finetune_supervised()
    trainer_mg_replication.add_hparams_to_writer()
    trainer_mg_replication.get_stats()


def resume_replicate_acs_results_fcnresnet18_my_cubes(run_nr, **kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]
    dataset_list = ["lidc_80_80_padded"]

    config = FineTuneConfig(
        data_dir="",
        task="REPLICATE_ACS_PAPER",
        self_supervised=False,
        supervised=True,
        model=kwargs_dict_["model"],
    )

    config.override_dirs(run_nr)

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
    trainer_mg_replication = Trainer(config, dataset)
    trainer_mg_replication.load_model(from_latest_checkpoint=True)
    trainer_mg_replication.finetune_supervised()
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
    split = kwargs_dict_.get("split", (0.8, 0.2, 0))
    mode = kwargs_dict_.get("mode", "")

    datasets_used_str = get_datasets_used_str(dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"])

    dataset = build_dataset(
        dataset_list=dataset_list,
        split=split,
        two_dimensional_data=kwargs_dict_["two_dimensional_data"],
        data_limit_2d=kwargs_dict_["data_limit_2d"],
    )

    # config = models_genesis_config(True, task="PRETRAIN_MG_FRAMEWORK{}".format(datasets_used_str))
    config = FineTuneConfig(
        data_dir="",
        task="PRETRAIN_MG_FRAMEWORK{}".format(datasets_used_str),
        self_supervised=True,
        supervised=False,
        model=kwargs_dict_["model"],
        extra_info_on_task_dir=False,
    )
    config.make_config_as_original_mg()
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

    datasets_used_str = get_datasets_used_str(dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"])
    # config = models_genesis_config(True, task="PRETRAIN_MG_FRAMEWORK{}".format(datasets_used_str))
    config = FineTuneConfig(
        data_dir="",
        task="PRETRAIN_MG_FRAMEWORK{}".format(datasets_used_str),
        self_supervised=True,
        supervised=False,
        model=kwargs_dict_["model"],
        extra_info_on_task_dir=False,
    )
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

    num_cv_folds = kwargs_dict_.get("num_cv_folds", None)
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()  # alphabetical, IF YOU DO NOT MAINTAIN ORDER A DIFFERENT TASK DIR IS CREATED FOR SAME DATASETS USED: eg: [lidc , brats] vs [brats, lids]
    split = kwargs_dict_.get("split", (0.8, 0.2, 0))
    mode = kwargs_dict_.get("mode", "")
    new_folder = kwargs_dict_["new_folder"]

    datasets_used_str = get_datasets_used_str(dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"])

    dataset = build_dataset(dataset_list=dataset_list, split=split, use_supervision_transforms=kwargs_dict_["use_supervision_transforms"])

    config = FineTuneConfig(
        data_dir="",
        task="FROM_PROVIDED_WEIGHTS{}".format(datasets_used_str)
        if kwargs_dict_["task_name"] is None
        else "{}{}".format(kwargs_dict_["task_name"], datasets_used_str),
        self_supervised=False,
        supervised=True,
        new_folder=new_folder,
    )
    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_from_provided_weights = True  # Redundant, just for logging purposes

    if num_cv_folds is not None:
        cv = get_cross_validator_object_of_task_dir(config.task_dir)
        if cv is None:
            if config.experiment_nr == 1:
                cv = CrossValidator(config, dataset, nr_splits=num_cv_folds)
                cv.override_dataset_files_with_splits()
                save_object(cv, "cross_validator", config.object_dir)
                print("RUN 1: Building cross validator")
            else:
                print("TOO LATE TO BRING CROSS VALIDATON IN")

        else:
            cv.set_dataset(dataset)
            cv.override_dataset_files_with_splits()
            # to "loose" used splits as they're popped and needs to be saved in run1 objects
            save_cross_validator_object_of_task_dir(cv, config.task_dir)

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
    new_folder = kwargs_dict_["new_folder"]

    datasets_used_str = get_datasets_used_str(dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"])
    config = FineTuneConfig(
        data_dir="",
        task="FROM_PROVIDED_WEIGHTS{}".format(datasets_used_str),
        self_supervised=False,
        supervised=True,
        new_folder=new_folder,
    )
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

    num_cv_folds = kwargs_dict_.get("num_cv_folds", None)
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()
    split = kwargs_dict_.get("split", (0.8, 0.2, 0))
    mode = kwargs_dict_.get("mode", "")
    datasets_used_str = get_datasets_used_str(dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"])
    new_folder = kwargs_dict_["new_folder"]

    dataset = build_dataset(dataset_list=dataset_list, split=split)

    config = FineTuneConfig(
        data_dir="",
        task="FROM_PROVIDED_WEIGHTS{}".format(datasets_used_str),
        self_supervised=True,
        supervised=True,
        model=kwargs_dict_["model"],
        new_folder=new_folder,
    )
    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_from_provided_weights = True  # Redundant, just for logging purposes

    if num_cv_folds is not None:
        cv = get_cross_validator_object_of_task_dir(config.task_dir)
        if cv is None:
            if config.experiment_nr == 1:
                cv = CrossValidator(config, dataset, nr_splits=num_cv_folds)
                cv.override_dataset_files_with_splits()
                save_object(cv, "cross_validator", config.object_dir)
                print("RUN 1: Building cross validator")
            else:
                print("TOO LATE TO BRING CROSS VALIDATON IN")

        else:
            cv.set_dataset(dataset)
            cv.override_dataset_files_with_splits()
            # to "loose" used splits as they're popped and needs to be saved in run1 objects
            save_cross_validator_object_of_task_dir(cv, config.task_dir)

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
    new_folder = kwargs_dict_["new_folder"]
    datasets_used_str = get_datasets_used_str(dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"])
    config = FineTuneConfig(
        data_dir="",
        task="FROM_PROVIDED_WEIGHTS{}".format(datasets_used_str),
        self_supervised=True,
        supervised=True,
        model=kwargs_dict_["model"],
        new_folder=new_folder,
    )
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


def use_model_weights_and_do_self_supervision(**kwargs):
    # pass it the directory of the task that the model you want to resume from is

    kwargs_dict_ = kwargs["kwargs_dict"]
    num_cv_folds = kwargs_dict_.get("num_cv_folds", None)
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()
    model_weights_dir = kwargs_dict_["directory"]
    split = kwargs_dict_.get("split", (0.8, 0.2, 0))
    mode = kwargs_dict_.get("mode", "")
    convert_acs = kwargs_dict_["convert_to_acs"]
    new_folder = kwargs_dict_["new_folder"]

    datasets_used_str = get_datasets_used_str(
        dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"], convert_to_acs=convert_acs
    )
    dataset = build_dataset(
        dataset_list=dataset_list,
        split=split,
        two_dimensional_data=kwargs_dict_["two_dimensional_data"],
        use_supervision_transforms=kwargs_dict_["use_supervision_transforms"],
    )

    config = FineTuneConfig(
        data_dir="",
        task="FROM_{}_DO_SS_ON_{}".format(model_weights_dir, datasets_used_str),
        self_supervised=True,
        supervised=False,
        model=kwargs_dict_["model"],
        new_folder=new_folder,
    )
    config.make_config_as_original_mg()
    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_from_specific_model = True  # Redundant, just for logging purposes

    if num_cv_folds is not None:
        cv = get_cross_validator_object_of_task_dir(config.task_dir)
        if cv is None:
            if config.experiment_nr == 1:
                cv = CrossValidator(config, dataset, nr_splits=num_cv_folds)
                cv.override_dataset_files_with_splits()
                save_object(cv, "cross_validator", config.object_dir)
                print("RUN 1: Building cross validator")
            else:
                print("TOO LATE TO BRING CROSS VALIDATON IN")

        else:
            cv.set_dataset(dataset)
            cv.override_dataset_files_with_splits()
            # to "loose" used splits as they're popped and needs to be saved in run1 objects
            save_cross_validator_object_of_task_dir(cv, config.task_dir)

    config.display()

    save_object(config, "config", config.object_dir)
    save_object(dataset, "dataset", config.object_dir)

    trainer = Trainer(config, dataset)
    trainer.load_model(from_directory=True, directory=model_weights_dir, convert_acs=convert_acs)
    trainer.finetune_self_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()


def resume_use_model_weights_and_do_self_supervision(run_nr: int, **kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()
    model_weights_dir = kwargs_dict_["directory"]  # to find the task dir to resume from
    mode = kwargs_dict_.get("mode", "")
    convert_acs = kwargs_dict_["convert_to_acs"]  # needs to be called on resume to find task dir

    datasets_used_str = get_datasets_used_str(
        dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"], convert_to_acs=convert_acs
    )

    config = FineTuneConfig(
        data_dir="",
        task="FROM_{}_DO_SS_ON_{}".format(model_weights_dir, datasets_used_str),
        self_supervised=True,
        supervised=False,
        model=kwargs_dict_["model"],
        new_folder=kwargs_dict_["new_folder"],
    )
    config.override_dirs(run_nr)

    if os.path.isfile(os.path.join(config.object_dir, "config.pkl")):
        config = load_object(os.path.join(config.object_dir, "config.pkl"))  #!
    else:
        raise FileNotFoundError(
            "Could not find CONFIG object pickle. Did you specify a valid run number? Path was {}".format(
                os.path.join(config.object_dir, "config.pkl")
            )
        )

    if os.path.isfile(os.path.join(config.object_dir, "dataset.pkl")):
        dataset = load_object(os.path.join(config.object_dir, "dataset.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find DATASET object pickle. Did you specify a valid run number?")

    replace_config_param_attributes(config, kwargs_dict_, ilegal=["model"])
    config.resume_ss = True
    config.display()

    trainer = Trainer(config, dataset)
    trainer.load_model(from_latest_checkpoint=True)  # if convert ACS the resume should already hae unet_acs as model
    trainer.finetune_self_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()


def use_model_weights_and_finetune_on_dataset_without_ss(**kwargs):
    # pass it the directory of the task that the model you want to resume from is

    kwargs_dict_ = kwargs["kwargs_dict"]

    num_cv_folds = kwargs_dict_.get("num_cv_folds", None)
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()
    model_weights_dir = kwargs_dict_["directory"]
    split = kwargs_dict_.get("split", (0.8, 0.2, 0))
    mode = kwargs_dict_.get("mode", "")
    convert_acs = kwargs_dict_["convert_to_acs"]
    new_folder = kwargs_dict_["new_folder"]
    pool_features = kwargs_dict_["pool_features"]

    datasets_used_str = get_datasets_used_str(
        dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"], convert_to_acs=convert_acs
    )
    dataset = build_dataset(
        dataset_list=dataset_list,
        split=split,
        two_dimensional_data=kwargs_dict_["two_dimensional_data"],
        use_supervision_transforms=kwargs_dict_["use_supervision_transforms"],
    )

    config = FineTuneConfig(
        data_dir="",
        task="FROM_{}_{}".format(model_weights_dir, datasets_used_str),
        self_supervised=False,
        supervised=True,
        model=kwargs_dict_["model"],
        new_folder=new_folder,
    )

    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_from_specific_model = True  # Redundant, just for logging purposes

    if num_cv_folds is not None:
        cv = get_cross_validator_object_of_task_dir(config.task_dir)
        if cv is None:
            if config.experiment_nr == 1:
                cv = CrossValidator(config, dataset, nr_splits=num_cv_folds)
                cv.override_dataset_files_with_splits()
                save_object(cv, "cross_validator", config.object_dir)
                print("RUN 1: Building cross validator")
            else:
                print("TOO LATE TO BRING CROSS VALIDATON IN")

        else:
            cv.set_dataset(dataset)
            cv.override_dataset_files_with_splits()
            # to "loose" used splits as they're popped and needs to be saved in run1 objects
            save_cross_validator_object_of_task_dir(cv, config.task_dir)

    config.display()

    save_object(config, "config", config.object_dir)
    save_object(dataset, "dataset", config.object_dir)

    trainer = Trainer(config, dataset)
    trainer.load_model(from_directory=True, directory=model_weights_dir, convert_acs=convert_acs, pool_features=pool_features)
    trainer.finetune_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()


def resume_use_model_weights_and_finetune_on_dataset_without_ss(run_nr: int, **kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()
    model_weights_dir = kwargs_dict_["directory"]  # to find the task dir to resume from
    mode = kwargs_dict_.get("mode", "")
    convert_acs = kwargs_dict_["convert_to_acs"]  # needs to be called on resume to find task dir
    new_folder = kwargs_dict_["new_folder"]

    datasets_used_str = get_datasets_used_str(
        dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"], convert_to_acs=convert_acs
    )

    config = FineTuneConfig(
        data_dir="",
        task="FROM_{}_{}".format(model_weights_dir, datasets_used_str),
        self_supervised=False,
        supervised=True,
        model=kwargs_dict_["model"],
        new_folder=kwargs_dict_["new_folder"],
    )
    config.override_dirs(run_nr)

    if os.path.isfile(os.path.join(config.object_dir, "config.pkl")):
        config = load_object(os.path.join(config.object_dir, "config.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find CONFIG object pickle. Did you specify a valid run number?")

    if os.path.isfile(os.path.join(config.object_dir, "dataset.pkl")):
        dataset = load_object(os.path.join(config.object_dir, "dataset.pkl"))  #!
    else:
        raise FileNotFoundError("Could not find DATASET object pickle. Did you specify a valid run number?")

    replace_config_param_attributes(config, kwargs_dict_, ilegal=["model"])
    config.resume_sup = True
    config.display()

    trainer = Trainer(config, dataset)
    trainer.load_model(from_latest_checkpoint=True)  # if convert ACS the resume should already hae unet_acs as model
    trainer.finetune_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()

    """
    ---
    """


def use_model_weights_and_finetune_on_dataset_with_ss(**kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]

    num_cv_folds = kwargs_dict_.get("num_cv_folds", None)
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()
    model_weights_dir = kwargs_dict_["directory"]
    split = kwargs_dict_.get("split", (0.8, 0.2, 0))
    mode = kwargs_dict_.get("mode", "")
    convert_acs = kwargs_dict_["convert_to_acs"]  # needs to be called on resume to find task dir

    datasets_used_str = get_datasets_used_str(
        dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"], convert_to_acs=convert_acs
    )
    dataset = build_dataset(dataset_list=dataset_list, split=split, two_dimensional_data=kwargs_dict_["two_dimensional_data"])

    config = FineTuneConfig(
        data_dir="",
        task="FROM_{}_{}".format(model_weights_dir, datasets_used_str),
        self_supervised=True,
        supervised=True,
        model=kwargs_dict_["model"],
    )
    replace_config_param_attributes(config, kwargs_dict_)
    config.resume_from_specific_model = True  # Redundant, just for logging purposes

    if num_cv_folds is not None:
        cv = get_cross_validator_object_of_task_dir(config.task_dir)
        if cv is None:
            if config.experiment_nr == 1:
                cv = CrossValidator(config, dataset, nr_splits=num_cv_folds)
                cv.override_dataset_files_with_splits()
                save_object(cv, "cross_validator", config.object_dir)
                print("RUN 1: Building cross validator")
            else:
                print("TOO LATE TO BRING CROSS VALIDATON IN")

        else:
            cv.set_dataset(dataset)
            cv.override_dataset_files_with_splits()
            # to "loose" used splits as they're popped and needs to be saved in run1 objects
            save_cross_validator_object_of_task_dir(cv, config.task_dir)

    config.display()

    save_object(config, "config", config.object_dir)
    save_object(dataset, "dataset", config.object_dir)

    trainer = Trainer(config, dataset)
    trainer.load_model(from_directory=True, directory=model_weights_dir, convert_acs=convert_acs)
    trainer.finetune_self_supervised()
    trainer.load_model(from_latest_improvement_ss=True)  # here it's already loading from the dir of the task
    trainer.finetune_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()


def resume_use_model_weights_and_finetune_on_dataset_with_ss(run_nr: int, **kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()
    model_weights_dir = kwargs_dict_["directory"]
    mode = kwargs_dict_.get("mode", "")
    convert_acs = kwargs_dict_["convert_to_acs"]  # needs to be called on resume to find task dir

    datasets_used_str = get_datasets_used_str(
        dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"], convert_to_acs=convert_acs
    )

    config = FineTuneConfig(
        data_dir="",
        task="FROM_{}_{}".format(model_weights_dir, datasets_used_str),
        self_supervised=True,
        supervised=True,
        model=kwargs_dict_["model"],
    )
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

    # acs resuming: if it's resuming config should already have unet_acs as model
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

    num_cv_folds = kwargs_dict_.get("num_cv_folds", None)
    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()  # alphabetical, IF YOU DO NOT MAINTAIN ORDER A DIFFERENT TASK DIR IS CREATED FOR SAME DATASETS USED: eg: [lidc , brats] vs [brats, lids]
    split = kwargs_dict_.get("split", (0.8, 0.2, 0))
    mode = kwargs_dict_.get("mode", "")
    make_acs_kernel_split_adaptive_to_input_dimensions = kwargs_dict_.get("make_acs_kernel_split_adaptive_to_input_dimensions", None)
    pool_features = kwargs_dict_["pool_features"]

    datasets_used_str = get_datasets_used_str(dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"])
    if make_acs_kernel_split_adaptive_to_input_dimensions is True:
        datasets_used_str = "_WITH_ADAPTIVE_ACS_KERNEL" + datasets_used_str

    dataset = build_dataset(
        dataset_list=dataset_list,
        split=split,
        two_dimensional_data=kwargs_dict_["two_dimensional_data"],
        use_supervision_transforms=kwargs_dict_["use_supervision_transforms"],
    )
    config = FineTuneConfig(
        data_dir="", task="FROM_SCRATCH{}".format(datasets_used_str), self_supervised=False, supervised=True, model=kwargs_dict_["model"]
    )
    replace_config_param_attributes(config, kwargs_dict_)
    config.from_scratch = True  # Redundant, just for logging purposes

    # TODO: move to function call
    if make_acs_kernel_split_adaptive_to_input_dimensions is True:
        x, y = dataset.get_train(batch_size=1)
        shape = x.shape[2:]
        total = 0
        for i in shape:
            total += i
        acs_kernel_split = tuple([float(i / total) for i in shape])
        dataset.reset()
    else:
        acs_kernel_split = None

    if num_cv_folds is not None:
        cv = get_cross_validator_object_of_task_dir(config.task_dir)
        if cv is None:
            if config.experiment_nr == 1:
                cv = CrossValidator(config, dataset, nr_splits=num_cv_folds)
                cv.override_dataset_files_with_splits()
                save_object(cv, "cross_validator", config.object_dir)
                print("RUN 1: Building cross validator")
            else:
                print("TOO LATE TO BRING CROSS VALIDATON IN")

        else:
            cv.set_dataset(dataset)
            cv.override_dataset_files_with_splits()
            # to "loose" used splits as they're popped and needs to be saved in run1 objects
            save_cross_validator_object_of_task_dir(cv, config.task_dir)

    config.display()

    save_object(config, "config", config.object_dir)
    save_object(dataset, "dataset", config.object_dir)

    trainer = Trainer(config, dataset)
    trainer.load_model(from_scratch=True, acs_kernel_split=acs_kernel_split, pool_features=pool_features)
    trainer.finetune_supervised()
    trainer.add_hparams_to_writer()
    trainer.get_stats()


def resume_train_from_scratch_on_dataset_no_ss(run_nr: int, **kwargs):

    kwargs_dict_ = kwargs["kwargs_dict"]

    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()  # alphabetical, IF YOU DO NOT MAINTAIN ORDER A DIFFERENT TASK DIR IS CREATED FOR SAME DATASETS USED: eg: [lidc , brats] vs [brats, lids]
    mode = kwargs_dict_.get("mode", "")
    datasets_used_str = get_datasets_used_str(dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"])
    config = FineTuneConfig(
        data_dir="", task="FROM_SCRATCH{}".format(datasets_used_str), self_supervised=False, supervised=True, model=kwargs_dict_["model"]
    )
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
    num_cv_folds = kwargs_dict_.get("num_cv_folds", None)

    dataset_list = kwargs_dict_["dataset"]
    dataset_list.sort()  # alphabetical, IF YOU DO NOT MAINTAIN ORDER A DIFFERENT TASK DIR IS CREATED FOR SAME DATASETS USED: eg: [lidc , brats] vs [brats, lids]
    split = kwargs_dict_.get("split", (0.8, 0.2, 0))
    mode = kwargs_dict_.get("mode", "")

    datasets_used_str = get_datasets_used_str(dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"])

    dataset = build_dataset(dataset_list=dataset_list, split=split, two_dimensional_data=kwargs_dict_["two_dimensional_data"])
    config = FineTuneConfig(
        data_dir="", task="FROM_SCRATCH{}".format(datasets_used_str), self_supervised=True, supervised=True, model=kwargs_dict_["model"]
    )

    replace_config_param_attributes(config, kwargs_dict_)
    config.from_scratch = True  # Redundant, just for logging purposes

    if num_cv_folds is not None:
        cv = get_cross_validator_object_of_task_dir(config.task_dir)
        if cv is None:
            if config.experiment_nr == 1:
                cv = CrossValidator(config, dataset, nr_splits=num_cv_folds)
                cv.override_dataset_files_with_splits()
                save_object(cv, "cross_validator", config.object_dir)
                print("RUN 1: Building cross validator")
            else:
                print("TOO LATE TO BRING CROSS VALIDATON IN")

        else:
            cv.set_dataset(dataset)
            cv.override_dataset_files_with_splits()
            # to "loose" used splits as they're popped and needs to be saved in run1 objects
            save_cross_validator_object_of_task_dir(cv, config.task_dir)

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
    datasets_used_str = get_datasets_used_str(dataset_list, mode, two_dim_data=kwargs_dict_["two_dimensional_data"])
    config = FineTuneConfig(
        data_dir="", task="FROM_SCRATCH{}".format(datasets_used_str), self_supervised=True, supervised=True, model=kwargs_dict_["model"]
    )
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
    task_name = kwargs_dict_["task_name"]
    enforce_test_again = kwargs_dict_["enforce_test_again"]
    mini_only = kwargs_dict_["mini_only"]
    full_only = kwargs_dict_["full_only"]
    task_dirs = get_task_dirs()
    # print("TASK DIRS ", task_dirs)
    for task_dir in task_dirs:
        if task_name is not None:
            if task_name not in task_dir:
                print("{} not in {}\n Continuing".format(task_name, task_dir))
                continue
        if not enforce_test_again:
            if not full_only:
                if task_dir_already_has_metric_dict_computed(task_dir) is True:
                    print("\n\n SKIPPED TESTING WEIGHTS FROM AS IS ALREADY COMPUTED: ", task_dir)
                    continue
            # only do full cubes after mini
            else:
                if task_dir_already_has_metric_dict_computed(task_dir) is False:
                    print("\n\n SKIPPED FULL CUBES TESTING WEIGHTS AS MINI IS NOT YET COMPUTED: ", task_dir)
                    continue
        if "FROM_PROVIDED_WEIGHTS_lidc_VNET_MG" in task_dir:
            if "new_folder" in task_dir:
                pass
            else:
                already_done = False
                split = task_dir.split("/")
                split[0] = "FROM_PROVIDED_WEIGHTS_SS_AND_SUP_lidc_VNET_MG"
                path = "/".join(split)
                if task_dir_already_has_metric_dict_computed(path):
                    already_done = True
                split = task_dir.split("/")
                split[0] = "FROM_PROVIDED_WEIGHTS_SUP_ONLY_lidc_VNET_MG"
                path = "/".join(split)
                if task_dir_already_has_metric_dict_computed(path):
                    already_done = True
                if already_done is True:
                    continue
        if "run_1_copy" in task_dir:
            continue
        if "UNET_ACS_CLS_ONLY" in task_dir:
            continue
        if "cellari_heart_sup_2D_UNET_2D" in task_dir or "cellari_heart_sup_10_192_2D_UNET_2D" in task_dir:
            continue
        config_object = get_config_object_of_task_dir(task_dir)
        if config_object is None:
            config_object = models_genesis_config(add_model_to_task=False)
            config_object.override_dirs(int(task_dir[-1]))
        if hasattr(config_object, "supervised") is False:
            print("SKIPPING, no supervised attribute in config: \n", task_dir)
            continue
        if config_object.supervised is False:
            print("SKIPPING, supervised is False in config: \n", task_dir)
            # not testing modules which have not been tuned for segmentation
            continue

        print("\n\n TESTING WEIGHTS FROM: ", task_dir)

        if ("FROM_PROVIDED_WEIGHTS_SUP_ONLY_lidc_VNET_MG" in config_object.model_path_save) or (
            "FROM_PROVIDED_WEIGHTS_SS_AND_SUP_lidc_VNET_MG" in config_object.model_path_save
        ):
            specific_weight_path_split = config_object.model_path_save.split("/")
            specific_weight_path_split[1] = "FROM_PROVIDED_WEIGHTS_lidc_VNET_MG"
            specific_weight_path = "/".join(specific_weight_path_split)
            config_object.model_path_save = specific_weight_path

        checkpoint = torch.load(
            os.path.join(config_object.model_path_save, "weights_sup.pt"),
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        if checkpoint.get("completed_sup", None) is not True:
            print("SKIPPING AS SUP IS NOT COMPLETED YET FOR {}".format(config_object.model_path_save))
            continue

        dataset_object = get_dataset_object_of_task_dir(task_dir)
        if dataset_object is None:
            x_train_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config_object.train_fold]
            x_val_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config_object.valid_fold]
            x_test_filenames = [
                "bat_32_s_64x64x32_" + str(i) + ".npy" for i in config_object.test_fold
            ]  # Dont know in what sense they use this for
            files = [x_train_filenames, x_val_filenames, x_test_filenames]
            dataset_object = Dataset(
                config_object.data_dir, train_val_test=(0.8, 0.2, 0), file_names=files
            )  # train_val_test is non relevant as is overwritten by files

        tester = Tester(config_object, dataset_object)
        if mini_only:
            tester.test_segmentation_mini()
        elif full_only:
            tester.test_segmentation_full()
        else:
            tester.test_segmentation_mini()
            tester.test_segmentation_full()


def extract_features(**kwargs):
    kwargs_dict_ = kwargs["kwargs_dict"]
    task_name = kwargs_dict_["task_name"]
    task_name_exact = kwargs_dict_["task_name_exact"]
    layer = kwargs_dict_["layer"]
    task_dirs = get_task_dirs()
    # print("TASK DIRS ", task_dirs)
    for task_dir in task_dirs:
        if task_name_exact is not None:
            if task_name_exact != task_dir:
                continue
        if task_name is not None:
            if task_name not in task_dir:
                print("{} not in {}\n Continuing".format(task_name, task_dir))
                continue

        if "run_1_copy" in task_dir:
            continue

        config_object = get_config_object_of_task_dir(task_dir)
        if config_object is None:
            raise ValueError

        print("\n\n EXTRACTING FEATURES FROM: ", task_dir)
        dataset_object = get_dataset_object_of_task_dir(task_dir)
        feature_extractor = FeatureExtractor(config_object, dataset_object)
        feature_extractor.extract_features(layer)


def plot_features(**kwargs):
    kwargs_dict_ = kwargs["kwargs_dict"]
    task_name = kwargs_dict_["task_name"]
    task_name_exact = kwargs_dict_["task_name_exact"]
    task_dirs = get_task_dirs()
    # print("TASK DIRS ", task_dirs)
    for task_dir in task_dirs:
        if task_name_exact is not None:
            if task_name_exact != task_dir:
                continue
        if task_name is not None:
            if task_name not in task_dir:
                print("{} not in {}\n Continuing".format(task_name, task_dir))
                continue

        if "run_1_copy" in task_dir:
            continue

        config_object = get_config_object_of_task_dir(task_dir)
        if config_object is None:
            raise ValueError

        print("\n\n EXTRACTING FEATURES FROM: ", task_dir)
        dataset_object = get_dataset_object_of_task_dir(task_dir)
        feature_extractor = FeatureExtractor(config_object, dataset_object)
        feature_extractor.plot_feature_maps_on_low_dimensional_space()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--command", required=True, dest="command", type=str)
    parser.add_argument("--run", required=False, dest="run", default=None, type=int)
    parser.add_argument("-d", "--dataset", nargs="+", required=False, dest="dataset", default=[])  # python arg.py -l 1234 2345 3456 4567
    parser.add_argument("--mode", required=False, dest="mode", default=None, type=str)
    parser.add_argument(
        "--directory",
        required=False,
        dest="directory",
        type=str,
        default=None,
        help="Path to Model weights folder. E.g: pretrained_weights/GENESIS_REPLICATION_PRETRAIN_MODEL/run_5",
    )
    parser.add_argument("--split", nargs="+", required=False, dest="tr_val_ts_split", default=None, type=str)

    parser.add_argument("-opt_ss", "--optimizer_ss", required=False, dest="optimizer_ss", type=str)
    parser.add_argument("-sch_ss", "--scheduler_ss", required=False, dest="scheduler_ss", type=str)
    parser.add_argument("-lr_ss", "--learning_rate_ss", required=False, dest="lr_ss", type=float)
    parser.add_argument("--batch_size_ss", required=False, dest="batch_size_ss", type=int)
    parser.add_argument("--patience_ss_terminate", required=False, dest="patience_ss_terminate", type=int)
    parser.add_argument("--patience_ss", required=False, dest="patience_ss", type=int)
    parser.add_argument("-opt_sup", "--optimizer_sup", required=False, dest="optimizer_sup", type=str)
    parser.add_argument("-sch_sup", "--scheduler_sup", required=False, dest="scheduler_sup", type=str)
    parser.add_argument("-lr_sup", "--learning_rate_sup", required=False, dest="lr_sup", type=float)
    parser.add_argument("--batch_size_sup", required=False, dest="batch_size_sup", type=int)
    parser.add_argument("--patience_sup_terminate", required=False, dest="patience_sup_terminate", type=int)
    parser.add_argument("--patience_sup", required=False, dest="patience_sup", type=int)
    parser.add_argument("--loss_function_sup", required=False, dest="loss_function_sup", type=str)
    parser.add_argument("--model", required=False, default="VNET_MG", dest="model", type=str)
    parser.add_argument("--task_name", required=False, dest="task_name", type=str, default=None)
    parser.add_argument("--task_name_exact", required=False, dest="task_name_exact", type=str, default=None)
    parser.add_argument("--num_cv_folds", dest="num_cv_folds", type=int, required=False, default=None)
    parser.add_argument("--two_dimensional_data", dest="two_dimensional_data", action="store_true", required=False)
    parser.add_argument("--convert_to_acs", dest="convert_to_acs", action="store_true", required=False)
    parser.add_argument("--new_folder", dest="new_folder", action="store_true", required=False)
    parser.add_argument("--use_supervision_transforms", dest="use_supervision_transforms", action="store_true", required=False)
    parser.add_argument(
        "--make_acs_kernel_split_adaptive_to_input_dimensions",
        dest="make_acs_kernel_split_adaptive_to_input_dimensions",
        action="store_true",
        required=False,
    )
    parser.add_argument("--data_limit_2d", dest="data_limit_2d", required=False, default=None, type=int)
    parser.add_argument("--enforce_test_again", dest="enforce_test_again", action="store_true", required=False)
    parser.add_argument("--mini_only", dest="mini_only", action="store_true", required=False)
    parser.add_argument("--full_only", dest="full_only", action="store_true", required=False)
    parser.add_argument("--pool_features", dest="pool_features", action="store_true", required=False)
    parser.add_argument("--layer", dest="layer", required=False, type=int, default=None)

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

    elif args.command == "replicate_acs_results_fcnresnet18_my_cubes":
        kwargs_dict = build_kwargs_dict(args, get_dataset=False, search_for_split=False)
        assert kwargs_dict["model"] is not None and kwargs_dict["model"].lower() != "vnet_mg"
        replicate_acs_results_fcnresnet18_my_cubes(kwargs_dict=kwargs_dict)

    elif args.command == "resume_replicate_acs_results_fcnresnet18_my_cubes":
        assert args.run is not None, "You have to specify which --run to resume (int)"
        kwargs_dict = build_kwargs_dict(args, get_dataset=False, search_for_split=False)
        assert kwargs_dict["model"] is not None and kwargs_dict["model"].lower() != "vnet_mg"
        resume_replicate_acs_results_fcnresnet18_my_cubes(run_nr=args.run, kwargs_dict=kwargs_dict)

    elif args.command == "replicate_acs_results_fcnresnet18_their_cubes":

        kwargs_dict = build_kwargs_dict(args, get_dataset=False, search_for_split=False)
        assert kwargs_dict["model"] is not None and kwargs_dict["model"].lower() != "vnet_mg"
        replicate_acs_results_fcnresnet18_their_cubes(kwargs_dict=kwargs_dict)

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
    elif args.command == "do_ss_from_model":
        kwargs_dict = build_kwargs_dict(args, get_dataset=True, search_for_split=True, get_directory=True)
        use_model_weights_and_do_self_supervision(kwargs_dict=kwargs_dict)

    elif args.command == "resume_do_ss_from_model":
        assert args.run is not None, "You have to specify which --run to resume (int)"
        kwargs_dict = build_kwargs_dict(args, get_dataset=True, get_directory=True)
        print("RESUMING SS FINETUNING FROM {} WEIGHTS SS FROM RUN {}".format(args.directory, args.run))
        print("DATASET: {} // MODE: {}".format(kwargs_dict["dataset"], args.mode))
        resume_use_model_weights_and_do_self_supervision(args.run, kwargs_dict=kwargs_dict)

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
        kwargs_dict = build_kwargs_dict(args, test=True, search_for_params=False)
        test(kwargs_dict=kwargs_dict)

    elif args.command == "extract_features":
        kwargs_dict = build_kwargs_dict(args, search_for_params=False)
        extract_features(kwargs_dict=kwargs_dict)

    elif args.command == "plot_features":
        kwargs_dict = build_kwargs_dict(args, search_for_params=False)
        plot_features(kwargs_dict=kwargs_dict)
    else:
        raise ValueError("Input a valid command")
