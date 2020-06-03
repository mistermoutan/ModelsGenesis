import os
import shutil

#FILE IS UNTOUCHED

def _get_task_dir(task):
    # get dir corresponding to next numerical experiment
    task_dir = task + "/run"
    experiment_nr = 1
    while os.path.isdir(os.path.join("runs/", task + "_" + str(experiment_nr) + "/")): #meaning the experiment has not been run
        experiment_nr += 1
        return task_dir + "_" + str(experiment_nr) + "/" 

class models_genesis_config:
    
    model = "Unet3D"
    suffix = "genesis_chest_ct"
    exp_name = model + "-" + suffix
    
    task = "GENESIS_REPLICATION"
    task_dir = _get_task_dir(task)
    stats_dir = os.path.join("stats/", task_dir)
    model_path_save = os.path.join("pretrained_weights/finetuning/", task_dir)
    summarywriter_dir = os.path.join("runs/", task_dir)
    
    # data
    data_dir = "/work1/s182312/luna16_extracted_cubes/scale_32"
    train_fold =[0,1,2,3,4]
    valid_fold=[5,6]
    test_fold=[7,8,9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    input_rows = 64
    input_cols = 64 
    input_deps = 32
    nb_class = 1
    
    # model pre-training
    self_supervised = True
    verbose = 1
    weights = None
    batch_size_ss = 6
    optimizer_ss = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch_ss = 10000
    patience_Ss = 50
    lr_ss = 1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    #in here just to conform with statistics module
    batch_size_sup = None
    optimizer_sup = None
    loss_function_sup = None #binary_cross_entropy
    nb_epoch_sup = None
    patience_sup = None
    lr_sup = None
    threshold =  None #above is considered part of mask

    # logs
    model_path = "pretrained_weights"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")