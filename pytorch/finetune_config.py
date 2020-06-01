import os
import shutil
from datetime import datetime
from utils import make_dir

class FineTuneConfig:
    
    def __init__(self, self_supervised: bool):
        
        self.self_supervised = self_supervised
    
        
        self.data_dir = "pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes"
        self.weights = "pretrained_weights/Genesis_Chest_CT.pt" #initial weights
        self.finetune_task = "TASK02_Heart_MRI"
        today_str = datetime.today().strftime('%Y-%m-%d-%H') + "h"
        self.stats_dir = os.path.join("stats/", self.finetune_task, today_str)
        self.model_path_save = os.path.join("pretrained_weights/finetuning/", self.finetune_task, today_str)
        make_dir(self.model_path_save)
            
        # self supervision finetuning (as done for building model genesis) before supervised finetuning
        if self.self_supervised:
            self.finetune_task = "With pre self-supervised" + self.finetune_task
            self.batch_size_ss = 6
            self.optimizer_ss = "sgd"
            self.loss_function_ss = "mse"
            self.nb_epoch_ss = 1000
            self.patience_ss = 50
            self.lr_ss = 5e-2  
            
            # image deformation for self supervision
            self.nonlinear_rate = 0.9
            self.paint_rate = 0.9
            self.outpaint_rate = 0.8
            self.inpaint_rate = 1.0 - self.outpaint_rate
            self.local_rate = 0.5
            self.flip_rate = 0.4
                
        # supervised finetuning
        self.batch_size_sup = 6
        self.optimizer_sup = "sgd"
        self.loss_function_sup = ""
        self.nb_epoch_sup = 10000
        self.patience_sup = 50
        self.lr_sup = 5e-4  
        #self.threshold =  0.5 #above is considered part of mask
            
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

