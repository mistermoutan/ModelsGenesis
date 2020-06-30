import os
import shutil
from utils import make_dir


class models_genesis_config:
    
    def __init__(self):
        
        self.model = "Unet3D"
        self.suffix = "genesis_chest_ct"
        self.exp_name = self.model + "-" + self.suffix

        self.task = "GENESIS_REPLICATION_PRETRAIN_MODEL"
        self.task_dir = self.get_task_dir(None)
        self.stats_dir = os.path.join("stats/", self.task_dir)
        self.model_path_save = os.path.join("pretrained_weights/", self.task_dir)
        self.summarywriter_dir = os.path.join("runs/", self.task_dir)
        make_dir(self.model_path_save)
        make_dir(self.stats_dir)

        #resume
        self.resume_ss = False
        self.resume_sup = False
        self.resume_from_provided_weights = False
        self.resume_from_ss_model = False
        
        # data
        #self.data_dir = "pytorch/datasets/luna16_cubes"
        #self.train_fold = [0]
        #self.valid_fold = [1]
        #self.test_fold = [2]
        self.data_dir = "/work1/s182312/luna16_extracted_cubes/scale_32"
        self.train_fold =[0,1,2,3,4]
        self.valid_fold=[5,6]
        self.test_fold=[7,8,9]
        
        self.hu_min = -1000.0
        self.hu_max = 1000.0
        self.scale = 32
        self.input_rows = 64
        self.input_cols = 64 
        self.input_deps = 32
        self.nb_class = 1

        # model pre-training
        self.self_supervised = True
        self.verbose = 1
        self.weights = "pretrained_weights/Genesis_Chest_CT.pt" #initial weights
        self.batch_size_ss = 6
        self.optimizer_ss = "sgd"
        self.workers = 10
        self.max_queue_size = self.workers * 4
        self.save_samples = "png"
        self.nb_epoch_ss = 10000
        self.patience_ss_terminate = 45
        self.patience_ss = self.patience_ss_terminate // 2
        self.loss_function_ss = "MSE" #binary_cross_entropy
        self.lr_ss = 1
        self.scheduler_ss = "StepLR" #"ReduceLROnPlateau"

        # image deformation
        self.nonlinear_rate = 0.9
        self.paint_rate = 0.9
        self.outpaint_rate = 0.8
        self.inpaint_rate = 1.0 - self.outpaint_rate
        self.local_rate = 0.5
        self.flip_rate = 0.4

        #in here just to conform with statistics module
        self.batch_size_sup = False
        self.optimizer_sup = False
        self.loss_function_sup = False #binary_cross_entropy
        self.nb_epoch_sup = False
        self.patience_sup = False
        self.patience_sup_terminate = False
        self.lr_sup = False
        self.scheduler_sup = False
        self.threshold = 0.5 #above is considered part of mask
        # logs
        self.model_path = "pretrained_weights"
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
    
    def get_task_dir(self, exp_nr=None):
        # get dir corresponding to next numerical experiment
        task_dir = self.task + "/run"
        if exp_nr is None:
            experiment_nr = 1
            while os.path.isdir(os.path.join("runs/", self.task + "/run_{}/".format(str(experiment_nr)))): #meaning the experiment has not been run
                experiment_nr += 1
        else:
            experiment_nr = exp_nr
            
        print("!==RUN_DIR==!", task_dir + "_" + str(experiment_nr) + "/" )
        return task_dir + "_" + str(experiment_nr) + "/" 

    def override_dirs(self, run_nr):
        
        task_dir = self.get_task_dir(exp_nr=run_nr)
        self.task_dir = task_dir
        self.stats_dir = os.path.join("stats/", task_dir)
        self.model_path_save = os.path.join("pretrained_weights/", task_dir)
        self.summarywriter_dir = os.path.join("runs/", task_dir)   

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")