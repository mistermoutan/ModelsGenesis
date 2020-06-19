import os
import shutil
from datetime import datetime
from utils import make_dir

class FineTuneConfig:
    
    def __init__(self, data_dir: str, task: str, self_supervised: bool, supervised:bool):
        
        self.self_supervised = self_supervised
        self.supervised = supervised
        self.task = task
        self.task_dir = self._get_task_dir(task)
        self.data_dir = data_dir #"pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes"
        self.weights = "pretrained_weights/Genesis_Chest_CT.pt" #initial weights
        #today_str = datetime.today().strftime('%Y-%m-%d-%H') + "h"
        self.stats_dir = os.path.join("stats/", self.task_dir)
        self.model_path_save = os.path.join("pretrained_weights/finetuning/", self.task_dir)
        self.summarywriter_dir = os.path.join("runs/", self.task_dir)
        make_dir(self.model_path_save)
        make_dir(self.stats_dir)
            
        # self supervision finetuning (as done for building model genesis) before supervised finetuning
        if self.self_supervised:
            self.batch_size_ss = 6
            self.optimizer_ss = "sgd"
            self.loss_function_ss = "mse"
            self.nb_epoch_ss = 1000
            self.patience_ss = 50
            self.lr_ss = 1e-3
            self.scheduler_ss = "steplr"
            # image deformation for self supervision
            self.nonlinear_rate = 0.9       
            self.paint_rate = 0.9
            self.outpaint_rate = 0.8
            self.inpaint_rate = 1.0 - self.outpaint_rate
            self.local_rate = 0.5
            self.flip_rate = 0.4
            
                
        # supervised finetuning
        self.batch_size_sup = 6
        self.optimizer_sup = "adam"
        self.beta1_sup = 0.9
        self.beta2_sup = 0.999
        self.eps_sup = 1e-8
        self.loss_function_sup = "dice" #binary_cross_entropy
        self.nb_epoch_sup = 10000
        self.patience_sup = 50
        self.lr_sup = 1e-3  
        self.scheduler_ss = "steplr"
        self.threshold =  0.5 #above is considered part of mask
                
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
        
    def _get_task_dir(self, task):
        # get dir corresponding to next numerical experiment
        task_dir = os.path.join(task, "with_self_supervised") if self.self_supervised else os.path.join(task, "only_supervised")
        experiment_nr = 1
        while os.path.isdir(os.path.join("runs/", task_dir + "/run_{}/".format(str(experiment_nr)))): #meaning the experiment has not been run
            experiment_nr += 1
        return task_dir + "_" + str(experiment_nr) + "/"
        
if __name__ == "__main__":
    
    from torch.utils.tensorboard import SummaryWriter
    for _ in range(3):
        config = FineTuneConfig(task="ModelGenesis/TASK02_Heart", data_dir="", self_supervised=True)
        #writer = SummaryWriter(os.path.join("runs", a.finetune_task_dir))
        print(os.path.join(config.model_path_save, "weights.pt"))
