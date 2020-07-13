import os
import shutil
from utils import make_dir


class models_genesis_config:
    def __init__(self, add_model_to_task: bool, task: str = None):

        self.model = "VNET_MG"
        self.suffix = "genesis_chest_ct"
        self.exp_name = self.model + "-" + self.suffix
        self.self_supervised = True

        # resume
        self.resume_ss = False
        self.resume_sup = False
        self.resume_from_provided_weights = False
        self.resume_from_ss_model = False
        self.resume_from_specific_model = False
        self.scratch = True

        self.task = "GENESIS_REPLICATION_PRETRAIN_MODEL" if task is None else task
        if add_model_to_task:
            self.task += "_{}".format(self.model)
        self.task_dir = self._get_task_dir(None)
        self.stats_dir = os.path.join("stats/", self.task_dir)
        self.model_path_save = os.path.join("pretrained_weights/", self.task_dir)
        self.summarywriter_dir = os.path.join("runs/", self.task_dir)
        self.object_dir = os.path.join("objects/", self.task_dir)
        make_dir(self.model_path_save)
        make_dir(self.stats_dir)
        make_dir(self.object_dir)

        self.weights = "pretrained_weights/Genesis_Chest_CT.pt"  # initial weights

        # data
        # self.data_dir = "pytorch/datasets/luna16_cubes"
        # self.train_fold = [0]
        # self.valid_fold = [0]
        # elf.test_fold = [0]
        self.data_dir = "/work1/s182312/luna16_extracted_cubes/scale_32"
        self.train_fold = [0, 1, 2, 3, 4]
        self.valid_fold = [5, 6]
        self.test_fold = [7, 8, 9]

        # image deformation
        self.nonlinear_rate = 0.9
        self.paint_rate = 0.9
        self.outpaint_rate = 0.8
        self.inpaint_rate = 1.0 - self.outpaint_rate
        self.local_rate = 0.5
        self.flip_rate = 0.4

        self.workers = 1
        self.max_queue_size = self.workers * 4
        self.save_samples = "png"

        # model pre-training

        self.batch_size_ss = 6
        self.optimizer_ss = "sgd"
        self.loss_function_ss = "MSE"
        self.nb_epoch_ss = 10000
        self.patience_ss_terminate = 50
        self.patience_ss = int(self.patience_ss_terminate * 0.4)
        self.lr_ss = 1
        self.scheduler_ss = "ReduceLROnPlateau"  # "ReduceLROnPlateau" or StepLr

        # in here just to conform with statistics and finetune module
        self.batch_size_sup = 6
        self.optimizer_sup = "adam"
        self.loss_function_sup = "dice"  # binary_cross_entropy
        self.nb_epoch_sup = 10000
        self.patience_sup_terminate = 50
        self.patience_sup = int(self.patience_sup_terminate * 0.4)
        self.lr_sup = 1e-3
        self.scheduler_sup = "steplr"
        self.beta1_sup = 0.9
        self.beta2_sup = 0.999
        self.eps_sup = 1e-8

        # logs
        self.model_path = "pretrained_weights"
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def _get_task_dir(self, exp_nr=None):
        # get dir corresponding to next numerical experiment
        task_dir = self.task + "/run"
        if exp_nr is None:
            experiment_nr = 1
            while os.path.isdir(os.path.join("runs/", self.task + "/run_{}/".format(str(experiment_nr)))):  # meaning the experiment has not been run
                experiment_nr += 1
        else:
            experiment_nr = exp_nr

        print("!==RUN_DIR==!", task_dir + "_" + str(experiment_nr) + "/")
        return task_dir + "_" + str(experiment_nr) + "/"

    def override_dirs(self, run_nr):

        task_dir = self._get_task_dir(exp_nr=run_nr)
        self.task_dir = task_dir
        self.stats_dir = os.path.join("stats/", task_dir)
        self.model_path_save = os.path.join("pretrained_weights/", task_dir)
        self.summarywriter_dir = os.path.join("runs/", task_dir)
        self.object_dir = os.path.join("objects/", self.task_dir)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
