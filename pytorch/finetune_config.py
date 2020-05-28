import os
import shutil
from datetime import datetime

class FineTuneConfig:
    
    def __init__(self, self_supervised=False):
        #perform supersvised task on new dataset before actually finetuning for segmentation
        self.self_supervised = self_supervised
    
    # data
    data_dir = "pytorch/datasets/Task02_Heart/imagesTr/extracted_cubes"
    weights = "pretrained_weights/Genesis_Chest_CT.pt" #initial weights
    finetune_task = "TASK02_Heart_MRI"
    today_str = datetime.today().strftime('%Y-%m-%d-%H') + "h"
    stats_dir = os.path.join("stats/", finetune_task, today_str)
    model_path_save = os.path.join("pretrained_weights/finetuning/", finetune_task, today_str)
    if not os.path.exists(model_path_save):
        os.makedirs(model_path_save)
        
    # model finetuning
    verbose = 1
    batch_size = 6
    optimizer = "sgd"
    nb_epoch = 1000
    patience = 50
    lr = 5e-4  
    self_supervised = False
    if self_supervised:
        finetune_task = "With pre self-supervised" + finetune_task

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
