import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import numpy as np
import time
from datetime import timedelta


from unet3d import UNet3D
from dataset import Dataset
from finetune_config import FineTuneConfig
from config import models_genesis_config
from stats import Statistics

from dice_loss import DiceLoss
from image_transformations import generate_pair

from dataset import Dataset

# TODO LOG:
#        CUBE DIMENSIONS
#        Scheduler
#
class Trainer:

    """ 
    Initializing model from scratch allows for:
        - Training on dataset from scratch with ModelGenesis self supervised framework
            
    Using pretrained weights as a starting point this class allows for finetuning the model by:
        - Using ModelGenesis self supervised framework (finetune_self_supervised)
        - Performing supervised task
    """

    def __init__(self, config: FineTuneConfig, dataset: Dataset):

        self.dataset = dataset
        self.config = config
        self.stats = Statistics(self.config, self.dataset)
        self.tb_writer = SummaryWriter(config.summarywriter_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_from_scratch_model_genesis_exact_replication(self):
        self.finetune_self_supervised() 
        
    def finetune_from_given_model_genesis_weights(self):
        raise NotImplementedError

    #@profile    
    def finetune_self_supervised(self):
        
        self._loadparams("ss")
        self.start_time = time.time()

        criterion = nn.MSELoss()

        if self.epoch_ss_check > 0:
            print("RESUMING SS TRAINING FROM EPOCH {} out of max {}".format(self.epoch_ss_check, self.config.nb_epoch_ss))
            print(self.best_loss_ss)
        else:
            print("STARTING SS TRAINING FROM SCRATCH")

        for self.epoch_ss_current in range(self.epoch_ss_check, self.config.nb_epoch_ss):

            self.stats.training_losses_ss = []
            self.stats.validation_losses_ss = []
            self.model.train()
            iteration = 0
            
            while True:  # go through all examples
                x, _ = self.dataset.get_train(batch_size=self.config.batch_size_ss, return_tensor=False)
                if x is None: break
                x_transform, y = generate_pair(x, self.config.batch_size_ss, self.config, make_tensors=True)
                x_transform, y = x_transform.float().to(self.device), y.float().to(self.device)
                pred = self.model(x_transform) 
                #print("INFERENCE SUCCESSFUL")
                loss = criterion(pred, y)
                self.optimizer_ss.zero_grad()
                loss.backward()
                self.optimizer_ss.step()
                self.tb_writer.add_scalar("Loss/train : Self Supervised", loss.item(), (self.epoch_ss_current + 1) * iteration)
                self.stats.training_losses_ss.append(loss.item())

                if (iteration + 1) % 1 == 0:
                    print(
                        "Epoch [{}/{}], iteration {}, Loss: {:.6f}".format(
                            self.epoch_ss_current + 1, self.config.nb_epoch_ss, iteration + 1, np.average(self.stats.training_losses_ss)
                        )
                    )
                    sys.stdout.flush()
                iteration += 1
                if iteration == 2: break

            with torch.no_grad():
                cnt = 0
                self.model.eval()
                x = 0
                while True:
                    x, _ = self.dataset.get_val(batch_size=self.config.batch_size_ss, return_tensor=False)
                    if x is None: break
                    x_transform, y = generate_pair(x, self.config.batch_size_ss, self.config, make_tensors=True)
                    x_transform, y = x_transform.float().to(self.device), y.float().to(self.device)
                    pred = self.model(x_transform)
                    loss = criterion(pred, y)
                    self.tb_writer.add_scalar("Loss/Validation : Self Supervised", loss.item(), (self.epoch_ss_current + 1) * iteration)
                    self.stats.validation_losses_ss.append(loss.item())
                    cnt += 1
                    if self.epoch_ss_current == 3: loss = -200 
                    if cnt == 2: break
            self.dataset.reset()
            self.scheduler_ss.step(self.epoch_ss_current)

            avg_training_loss_of_epoch = np.average(self.stats.training_losses_ss)
            self.tb_writer.add_scalar("Avg Loss Epoch/Training : Self Supervised", avg_training_loss_of_epoch, self.epoch_ss_current + 1)
            avg_validation_loss_of_epoch = np.average(self.stats.validation_losses_ss)
            self.tb_writer.add_scalar("Avg Loss Epoch/Validation : Self Supervised", avg_validation_loss_of_epoch, self.epoch_ss_current + 1)

            self.stats.avg_training_loss_per_epoch_ss.append(avg_training_loss_of_epoch)
            self.stats.avg_validation_loss_per_epoch_ss.append(avg_validation_loss_of_epoch)
            self.stats.iterations_ss.append(iteration)

            print("###### SELF SUPERVISED#######")
            
            print(
                "Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(
                    self.epoch_ss_current + 1, avg_validation_loss_of_epoch, avg_training_loss_of_epoch
                )
            )
            if avg_validation_loss_of_epoch < self.best_loss_ss:
                print("Validation loss decreases from {:.4f} to {:.4f}".format(self.best_loss_ss, avg_validation_loss_of_epoch))
                self.best_loss_ss = avg_validation_loss_of_epoch
                self.num_epoch_no_improvement_ss = 0
                self._save_model("ss")
                print("Model Saved in {}".format(self.config.model_path_save))
            else:
                print("Validation loss did not decrease from {:.4f}, num_epoch_no_improvement {}".format(self.best_loss_ss, self.num_epoch_no_improvement_ss + 1))
                self.num_epoch_no_improvement_ss += 1
            if self.num_epoch_no_improvement_ss >= self.config.patience_ss:
                print("Early Stopping SS")
                self.stats.stopped_early_ss = True
                break
            sys.stdout.flush()
            self.tb_writer.add_scalar("Num epochs w/ no improvement: Self Supervised", self.num_epoch_no_improvement_ss, self.epoch_ss_current + 1)
            
        self.ss_timedelta = timedelta(seconds= time.time() - self.start_time)
        print("FINISHED TRAINING SS")

    def finetune_supervised(self):
        
        self._loadparams("sup")
        self.start_time = time.time()

        if self.config.loss_function_sup == "binary_cross_entropy":
            criterion = nn.BCELoss()  # #model outputs sigmoid so no use of BCEwithLogits
        elif self.config.loss_function_sup == "dice":
            criterion = DiceLoss.dice_loss
        elif self.config.loss_function_sup == "mix_dice_bce":
            # https://discuss.pytorch.org/t/dice-loss-cross-entropy/53194
            raise NotImplementedError

        if self.epoch_sup_check > 0:
            print("RESUMING SUP TRAINING FROM EPOCH {} out of max {}".format(self.epoch_sup_check, self.config.nb_epoch_sup))
        else:
            print("STARTING SUP TRAINING FROM SCRATCH")
        
        for self.epoch_sup_current in range(self.epoch_sup_check, self.config.nb_epoch_sup):
            self.stats.training_losses_sup = []
            self.stats.validation_losses_sup = []
            self.model.train()
            iteration = 0
            while True:  # go through all examples
                x, y = self.dataset.get_train(batch_size=self.config.batch_size_sup)
                if x is None: break
                x, y = x.float().to(self.device), y.float().to(self.device)
                pred = self.model(x)
                loss = criterion(pred, y)
                self.optimizer_sup.zero_grad()
                loss.backward()
                self.optimizer_sup.step()
                self.tb_writer.add_scalar("Loss/train : Supervised", loss.item(), (self.epoch_sup_current + 1) * iteration)
                self.stats.training_losses_sup.append(loss.item())

                if (iteration + 1) % 1 == 0:
                    print(
                        "Epoch [{}/{}], iteration {}, Loss: {:.6f}".format(
                            self.epoch_sup_current + 1, self.config.nb_epoch_sup, iteration + 1, np.average(self.stats.training_losses_sup)
                        )
                    )
                    sys.stdout.flush()
                iteration += 1

            with torch.no_grad():
                self.model.eval()
                while True:
                    x, y = self.dataset.get_val(batch_size=self.config.batch_size_sup)
                    if x is None: break
                    x, y = x.float().to(self.device), y.float().to(self.device)
                    pred = self.model(x)
                    loss = criterion(pred, y)
                    self.tb_writer.add_scalar("Loss/Validation : Supervised", loss.item(), (self.epoch_sup_current + 1) * iteration)
                    self.stats.validation_losses_sup.append(loss.item())

            self.dataset.reset()
            self.scheduler_sup.step(self.epoch_sup_current)
            avg_training_loss_of_epoch = np.average(self.stats.training_losses_sup)
            self.tb_writer.add_scalar("Avg Loss Epoch/Training : Supervised", avg_training_loss_of_epoch, self.epoch_sup_current + 1)
            avg_validation_loss_of_epoch = np.average(self.stats.validation_losses_sup)
            self.tb_writer.add_scalar("Avg Loss Epoch/Validation : Supervised", avg_validation_loss_of_epoch, self.epoch_sup_current + 1)

            self.stats.avg_training_loss_per_epoch_sup.append(avg_training_loss_of_epoch)
            self.stats.avg_validation_loss_per_epoch_sup.append(avg_validation_loss_of_epoch)
            self.stats.iterations_sup.append(iteration)

            avg_training_loss_of_epoch = np.average(self.stats.training_losses_sup)
            avg_validation_loss_of_epoch = np.average(self.stats.validation_losseup_sup)

            print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(self.epoch_sup_current + 1, avg_validation_loss_of_epoch, avg_training_loss_of_epoch))
            if avg_validation_loss_of_epoch < self.best_loss_sup:
                print("Validation loss decreases from {:.4f} to {:.4f}".format(self.best_loss_sup, avg_validation_loss_of_epoch))
                self.best_loss_sup = avg_validation_loss_of_epoch
                num_epoch_no_improvement = 0
                # save model
                self._save_model("sup")
                print("Model Saved")
            else:
                print("Validation loss did not decrease from {:.4f}, num_epoch_no_improvement {}".format(self.best_loss_sup, self.num_epoch_no_improvement_sup + 1))
                self.num_epoch_no_improvement_sup += 1
            if self.num_epoch_no_improvement_sup >= self.config.patience_sup:
                print("Early Stopping SUP")
                self.stats.stopped_early_sup = True
                break
            sys.stdout.flush()
            self.tb_writer.add_scalar("Num epochs w/ no improvement: Supervised", self.num_epoch_no_improvement_sup, self.epoch_sup_current + 1)

        print("FINISHED TRAINING SUP")
        self.sup_timedelta = timedelta(seconds= time.time() - self.start_time)
        
    def test(self, test_dataset):
        from evaluate import Tester
        self.tester = Tester(self.model, self.config)
        self.tester.test_segmentation(test_dataset)

    def add_hparams_to_writer(self):
        
        hpa_dict = {   
                "cube_dimensions": self.dataset.cube_dimensions,
                "initial_lr_ss": self.config.lr_ss,
                "loss_ss": self.config.loss_function_ss,
                "optimizer_ss": self.config.optimizer_ss,
                "scheduler_ss": self.config.scheduler_ss,
                "batch_size_ss": self.config.batch_size_ss,
                "nr_epochs_ss": self.config.nb_epoch_ss,
                "initial_lr_sup": self.config.lr_sup,
                "loss_sup": self.config.loss_function_sup,
                "optimizer_sup": self.config.optimizer_sup,
                "scheduler_sup": self.config.scheduler_sup,
                "batch_size_sup": self.config.batch_size_sup,
                "nr_epochs_sup": self.config.nb_epoch_sup
            }
               
        self.ss_timedelta = 0 if (not hasattr(self, "ss_timedelta")) else (self.ss_timedelta.seconds // 60 % 60)
        self.sup_timedelta = 0 if (not hasattr(self, "sup_timedelta")) else (self.sup_timedelta.seconds // 60 % 60)
        self.stats.last_avg_training_loss_per_epoch_ss = 0 if not self.stats.avg_training_loss_per_epoch_ss else self.stats.avg_training_loss_per_epoch_ss[-1]
        self.stats.last_avg_validation_loss_per_epoch_ss = 0 if not self.stats.avg_validation_loss_per_epoch_ss else self.stats.avg_validation_loss_per_epoch_ss[-1]
        self.stats.last_avg_training_loss_per_epoch_sup = 0 if not self.stats.avg_training_loss_per_epoch_sup else self.stats.avg_training_loss_per_epoch_sup[-1]
        self.stats.last_avg_validation_loss_per_epoch_sup = 0 if not self.stats.avg_validation_loss_per_epoch_sup else self.stats.avg_validation_loss_per_epoch_sup[-1]
        
        met_dict = {
                    "final_train_loss_ss": self.stats.last_avg_training_loss_per_epoch_ss,
                    "final_val_loss_ss": self.stats.last_avg_validation_loss_per_epoch_ss,
                    "stopped_early_ss": self.stats.stopped_early_ss,
                    "training_time_ss": self.ss_timedelta,
                    "final_train_loss_sup": self.stats.last_avg_training_loss_per_epoch_sup,
                    "final_val_loss_sup": self.stats.last_avg_validation_loss_per_epoch_sup,
                    "stopped_early_sup": self.stats.stopped_early_sup,
                    "training_time_sup": self.sup_timedelta
                    }
        
        if hasattr(self, "tester"):
            if isinstance(self.tester.dice, float):
                met_dict.update({"test_dice": self.tester.dice, "test_jaccard": self.tester.jaccard})
            elif isinstance(self.tester.dice, list):
                met_dict.update({"test_dice_{}".format(str(i)): dice for i, dice in enumerate(self.tester.dice)})
                met_dict.update({"test_jaccard_{}".format(str(i)): jacd for i, jacd in enumerate(self.tester.jaccard)})
        
        self.tb_writer.add_hparams(hparam_dict=hpa_dict, metric_dict=met_dict)
        self.tb_writer.flush()
        self.tb_writer.close()
        
        
    def load_model(self, **kwargs):


        self.model = UNet3D()
        
        from_latest_checkpoint = kwargs.get("from_latest_checkpoint", False)
        from_weights = kwargs.get("from_weights", False)
        from_scratch = kwargs.get("from_scratch", False)
        
        if from_latest_checkpoint:
            if self.config.resume_ss:
                weight_dir = os.path.join(self.config.model_path_save, "weights_ss.pt") if from_latest_checkpoint and os.path.isfile(os.path.join(self.config.model_path_save, "weights_ss.pt")) else None
            elif self.config.resume_sup:
                weight_dir = os.path.join(self.config.model_path_save, "weights_sup.pt") if from_latest_checkpoint and os.path.isfile(os.path.join(self.config.model_path_save, "weights_sup.pt")) else None
        
        if from_weights:
                weight_dir = self.config.weights if os.path.isfile(self.config.weights) else None
                
        if from_scratch:
            weight_dir = None
        
        if weight_dir is None: print("Loading Model with random weights")
        
        else:          
            print("LOADING WEIGHTS FROM {}".format(weight_dir))
            checkpoint = torch.load(weight_dir, map_location=self.device)
            if self.config.resume_ss:
                state_dict = checkpoint["model_state_dict_ss"]
            if self.config.resume_sup:
                state_dict = checkpoint["model_state_dict_sup"]
            if from_weights:
                state_dict = checkpoint["state_dict"]
                
            unParalled_state_dict = {}
            for key in state_dict.keys():
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            self.model.load_state_dict(unParalled_state_dict)  ### TODO: WHY IS THIS UNPARALLELED NECESSARY?

        self.model.to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(torch.cuda.device_count())])
    
    
    def _save_model(self, phase:str):
        """
        Args:
            phase (str): ["ss" or "sup"]
        """
        
        assert phase == "ss" or phase == "sup"
        
        if phase == "ss":
            torch.save(
                {"epoch_ss": self.epoch_ss_current + 1, "model_state_dict_ss": self.model.state_dict(), "optimizer_state_dict_ss": self.optimizer_ss.state_dict(),
                    "scheduler_state_dict_ss": self.scheduler_ss.state_dict(), "num_epoch_no_improvement_ss": self.num_epoch_no_improvement_ss,
                    "best_loss_ss": self.best_loss_ss},
                os.path.join(self.config.model_path_save, "weights_ss.pt"),
            )
            
        elif phase == "sup":
            torch.save(
                {"epoch_sup": self.epoch_sup_current + 1, "model_state_dict_sup": self.model.state_dict(), "optimizer_state_dict_sup": self.optimizer_sup.state_dict(),
                    "scheduler_state_dict_sup": self.scheduler_sup.state_dict(),  "num_epoch_no_improvement_sup": self.num_epoch_no_improvement_sup,
                    "best_loss_sup": self.best_loss_sup},
                os.path.join(self.config.model_path_save, "weights_sup.pt"),
            )
            
    def _loadparams(self, phase:str):
    
        if phase=="ss":
            
            if self.config.optimizer_ss == "sgd":
                self.optimizer_ss = torch.optim.SGD(self.model.parameters(), self.config.lr_ss, momentum=0.9, weight_decay=0.0, nesterov=False)
            elif self.config.optimizer_ss == "adam":
                self.optimizer_ss = torch.optim.Adam(self.model.parameters(), self.config.lr_ss)
                
            if self.config.scheduler_ss == "ReduceLROnPlateau" or "ReduceLROnPlateau".lower():
                self.scheduler_ss = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_ss)
            elif self.config.scheduler_ss == "StepLR" or "StepLR".lower():
                self.scheduler_ss = torch.optim.lr_scheduler.StepLR(self.optimizer_ss, step_size=int(self.config.patience_ss * 0.8), gamma=0.5)
                
            if self.config.resume_ss:
                weight_dir = None
                if os.path.isfile(os.path.join(self.config.model_path_save, "weights_ss.pt")):
                    print("RESUMING FROM SS CHECKPOINT")
                    weight_dir = os.path.join(self.config.model_path_save, "weights_ss.pt")
                    
                elif os.path.isfile(os.path.join(self.config.model_path_save, "weights.pt")):
                    print("RESUMING FROM ORIGINAL GIVEN PRETRAINED WEIGHTTS")
                    weight_dir = os.path.join(self.config.weights, "weights.pt")

                if weight_dir is None: raise FileNotFoundError("Trying to resume from non existent checkpoint")
                    
                print("LOADING PARAMS FROM {}".format(weight_dir))
                checkpoint = torch.load(weight_dir, map_location=self.device)
                self.optimizer_ss.load_state_dict(checkpoint["optimizer_state_dict_ss"])
                self.scheduler_ss.load_state_dict(checkpoint["scheduler_state_dict_ss"])
            else:
                checkpoint = {}
                
            self.num_epoch_no_improvement_ss = checkpoint.get("num_epoch_no_improvement_ss", 0)
            self.stats.training_losses_ss = checkpoint.get("training_losses_ss", [])
            self.stats.validation_losses_ss = checkpoint.get("validation_losses_ss", [])
            self.epoch_ss_check = checkpoint.get("epoch_ss", 0)
            self.best_loss_ss = checkpoint.get("best_loss_ss", 10000000000)
    
        elif phase == "sup":

            if self.config.optimizer_sup == "sgd":
                self.optimizer_sup = torch.optim.SGD(self.model.parameters(), self.config.lr_sup, momentum=0.9, weight_decay=0.0, nesterov=False)
            elif self.config.optimizer_sup == "adam":
                self.optimizer_sup = torch.optim.Adam(
                    self.model.parameters(), self.config.lr_sup, betas=(self.config.beta1_sup, self.config.beta2_sup), eps=self.config.eps_sup
                )
            
            self.scheduler_sup = torch.optim.lr_scheduler.StepLR(self.optimizer_sup, step_size=int(self.config.patience_sup * 0.8), gamma=0.5)
                
            if self.config.resume_sup:
                weight_dir = None
                if os.path.isfile(os.path.join(self.config.model_path_save, "weights_sup.pt")):
                    weight_dir = os.path.join(self.config.model_path_save, "weights_sup.pt")
                elif os.path.isfile(os.path.join(self.config.model_path_save, "weights_ss.pt")):
                    weight_dir = os.path.join(self.config.model_path_save, "weights_ss.pt")
                    print("LOADING FROM PREVIOUS SS to perform SUP in dir:{}",format(weight_dir))
                else:
                    weight_dir = os.path.join(self.config.weights, "weights.pt")

                if weight_dir is None: raise FileNotFoundError("Trying to resume from non existent checkpoint")
                    
                print("LOADING PARAMS FROM {}".format(weight_dir))
                checkpoint = torch.load(weight_dir, map_location=self.device)
                self.optimizer_sup.load_state_dict(checkpoint["optimizer_state_dict_sup"])
                self.scheduler_sup.load_state_dict(checkpoint["scheduler_state_dict_sup"])
            else:
                checkpoint = {}
                
            self.num_epoch_no_improvement_sup = checkpoint.get("num_epoch_no_improvement_sup", 0)
            self.stats.training_losses_sup = checkpoint.get("training_losses_sup", [])
            self.stats.validation_losses_sup = checkpoint.get("validation_losses_sup", [])
            self.epoch_sup_check = checkpoint.get("epoch_sup", 0)
            self.best_loss_sup = checkpoint.get("best_loss_sup", 10000000000)

    def get_stats(self):
        self.stats.get_statistics()


if __name__ == "__main__":
    pass
    #config = models_genesis_config()
    #dataset = Dataset(config.data_dir, train_val_test=(0.8, 0.2, 0)) # train_val_test is non relevant as will ve overwritten after
    #dataset.x_train_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.train_fold]
    #dataset.x_val_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.valid_fold]
    #dataset.x_test_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.test_fold] #Dont know in what sense they use this for
    #trainer_mg_replication = Trainer(config, dataset)
    #trainer_mg_replication.train_from_scratch_model_model_genesis_exact_replication()
    #trainer_mg_replication.add_hparams_to_writer()
    #trainer_mg_replication.get_stats()  

