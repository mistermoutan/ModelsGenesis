import os
import shutil
from datetime import datetime
from utils import make_dir


class FineTuneConfig:
    def __init__(
        self,
        data_dir: str,
        task: str,
        self_supervised: bool,
        supervised: bool,
        model="VNET_MG",
        extra_info_on_task_dir=True,
        new_folder=False,
    ):

        self.model = model
        self.self_supervised = self_supervised  # only for stats module
        self.supervised = supervised
        self.extra_info_on_task_dir = extra_info_on_task_dir
        self.new_folder = new_folder  # to repeat experiments put them on remake folder for separation, used in obtaining task_dir

        # resume - The ones to interact with manually in functions of main, rest will come from kwargs
        self.resume_ss = False
        self.resume_sup = False
        self.resume_from_provided_weights = False
        self.resume_from_ss_model = False
        self.resume_from_specific_model = False
        self.from_scratch = False

        self.workers = 1

        self.task = task + "_{}".format(self.model.upper())
        self.task_dir = self._get_task_dir(None)
        self.stats_dir = os.path.join("stats/", self.task_dir)
        self.model_path_save = os.path.join("pretrained_weights/", self.task_dir)
        self.summarywriter_dir = os.path.join("runs/", self.task_dir)
        self.object_dir = os.path.join("objects/", self.task_dir)
        make_dir(self.model_path_save)
        make_dir(self.stats_dir)
        make_dir(self.object_dir)

        self.weights = "pretrained_weights/Genesis_Chest_CT.pt"  # provided weights

        self.data_dir = data_dir  # "pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes"

        # image deformation for self supervision
        self.nonlinear_rate = 0.9
        self.paint_rate = 0.9
        self.outpaint_rate = 0.8
        self.inpaint_rate = 1.0 - self.outpaint_rate
        self.local_rate = 0.5
        self.flip_rate = 0.4

        # self supervision finetuning
        self.batch_size_ss = 6
        self.optimizer_ss = "sgd"
        self.loss_function_ss = "MSE"
        self.nb_epoch_ss = 1000
        self.patience_ss_terminate = 30
        self.patience_ss = int(self.patience_ss_terminate * 0.7)
        self.lr_ss = 1e-3
        self.scheduler_ss = "StepLR"

        # supervised finetuning
        self.batch_size_sup = 6
        self.optimizer_sup = "adam"
        self.loss_function_sup = "dice"  # binary_cross_entropy
        self.nb_epoch_sup = 10000
        self.patience_sup_terminate = 50
        self.patience_sup = int(self.patience_sup_terminate * 0.7)
        self.lr_sup = 1e-3
        self.scheduler_sup = "steplr"
        self.beta1_sup = 0.9
        self.beta2_sup = 0.999
        self.eps_sup = 1e-8

        # testing
        self.threshold = 0.5  # above is considered part of mask
        self.mode = "DEFINE ME"

    def make_config_as_original_mg(self):

        self.from_scratch = False
        self.batch_size_ss = 6
        self.optimizer_ss = "sgd"
        self.loss_function_ss = "MSE"
        self.nb_epoch_ss = 10000
        self.patience_ss_terminate = 50
        self.patience_ss = int(self.patience_ss_terminate * 0.4)
        self.lr_ss = float(1)
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

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def _get_task_dir(self, exp_nr=None):
        # get dir corresponding to next numerical experiment
        if self.extra_info_on_task_dir is True:
            task_dir = self.task + "/with_self_supervised/run" if self.self_supervised else self.task + "/only_supervised/run"
        else:
            task_dir = self.task + "/run" if self.self_supervised else self.task + "/run"
        if exp_nr is None:
            self.experiment_nr = 1
            while os.path.isdir(
                os.path.join("runs/", task_dir[:-4] + "/run_{}/".format(str(self.experiment_nr)))
            ):  # meaning the experiment has not been run
                self.experiment_nr += 1
        else:
            self.experiment_nr = exp_nr

        if self.new_folder is False:
            print("!==RUN_DIR==!", task_dir + "_" + str(self.experiment_nr) + "/")
            return task_dir + "_" + str(self.experiment_nr) + "/"
        else:
            print("!==RUN_DIR==!", "new_folder/" + task_dir + "_" + str(self.experiment_nr) + "/")
            return "new_folder/" + task_dir + "_" + str(self.experiment_nr) + "/"

    def override_dirs(self, run_nr):

        task_dir = self._get_task_dir(exp_nr=run_nr)
        self.task_dir = task_dir
        self.stats_dir = os.path.join("stats/", task_dir)
        self.model_path_save = os.path.join("pretrained_weights/", task_dir)
        self.summarywriter_dir = os.path.join("runs/", task_dir)
        self.object_dir = os.path.join("objects/", self.task_dir)
