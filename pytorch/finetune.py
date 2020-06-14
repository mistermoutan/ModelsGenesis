import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import numpy as np
from memory_profiler import profile

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
        self._loadmodel()

    def train_from_scratch_model_model_genesis_exact_replication(self):

        self._loadmodel(from_weights=False)
        self.finetune_self_supervised()

    def finetune_self_supervised(self):

        if self.config.optimizer_ss == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), self.config.lr_ss, momentum=0.9, weight_decay=0.0, nesterov=False)
        elif self.config.optimizer_ss == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), self.config.lr_ss)

        if self.config.scheduler_ss == "ReduceLROnPlateau" or "ReduceLROnPlateau".lower():
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        elif self.config.scheduler_ss == "StepLR" or "StepLR".lower():
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(self.config.patience_ss * 0.8), gamma=0.5)

        best_loss = 10000000000
        num_epoch_no_improvement = 0
        criterion = nn.MSELoss()

        for epoch in range(self.config.nb_epoch_ss):
            self.stats.training_losses_ss = []
            self.stats.validation_losses_ss = []
            self.model.train()
            iteration = 0
            while True:  # go through all examples
                x, _ = self.dataset.get_train(batch_size=self.config.batch_size_ss, return_tensor=False)
                if x is None:
                    break
                x_transform, y = generate_pair(x, self.config.batch_size_ss, self.config, make_tensors=True)
                x_transform, y = x_transform.float().to(self.device), y.float().to(self.device)
                pred = self.model(x_transform)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.tb_writer.add_scalar("Loss/train : Self Supervised", loss, (epoch + 1) * iteration)
                self.stats.training_losses_ss.append(loss.item())

                if (iteration + 1) % 5 == 0:
                    print(
                        "Epoch [{}/{}], iteration {}, Loss: {:.6f}".format(
                            epoch + 1, self.config.nb_epoch_ss, iteration + 1, np.average(self.stats.training_losses_ss)
                        )
                    )
                    sys.stdout.flush()
                iteration += 1

            with torch.no_grad():
                self.model.eval()
                x = 0
                while True:
                    x, _ = self.dataset.get_val(batch_size=self.config.batch_size_ss, return_tensor=False)
                    if x is None:
                        break
                    x_transform, y = generate_pair(x, self.config.batch_size_ss, self.config, make_tensors=True)
                    x_transform, y = x_transform.float().to(self.device), y.float().to(self.device)
                    pred = self.model(x_transform)
                    loss = criterion(pred, y)
                    self.tb_writer.add_scalar("Loss/Validation : Self Supervised", loss, (epoch + 1) * iteration)
                    self.stats.validation_losses_ss.append(loss.item())

            self.dataset.reset()
            scheduler.step(epoch)

            avg_training_loss_of_epoch = np.average(self.stats.training_losses_ss)
            self.tb_writer.add_scalar("Avg Loss Epoch/Training : Self Supervised", avg_training_loss_of_epoch, epoch + 1)
            avg_validation_loss_of_epoch = np.average(self.stats.validation_losses_ss)
            self.tb_writer.add_scalar("Avg Loss Epoch/Validation : Self Supervised", avg_validation_loss_of_epoch, epoch + 1)

            self.stats.avg_training_loss_per_epoch_ss.append(avg_training_loss_of_epoch)
            self.stats.avg_validation_loss_per_epoch_ss.append(avg_validation_loss_of_epoch)
            self.stats.iterations_ss.append(iteration)

            print("###### SELF SUPERVISED#######")
            print(
                "Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(
                    epoch + 1, avg_validation_loss_of_epoch, avg_training_loss_of_epoch
                )
            )
            if avg_validation_loss_of_epoch < best_loss:
                print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, avg_validation_loss_of_epoch))
                best_loss = avg_validation_loss_of_epoch
                num_epoch_no_improvement = 0
                # save model
                torch.save(
                    {"epoch": epoch + 1, "state_dict": self.model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                    os.path.join(self.config.model_path_save, "weights.pt"),
                )
                print("Model Saved")
            else:
                print("Validation loss did not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss, num_epoch_no_improvement))
                num_epoch_no_improvement += 1
            if num_epoch_no_improvement >= self.config.patience_ss:
                print("Early Stopping SS")
                self.stats.stopped_early_ss = True
                break
            sys.stdout.flush()
            self.tb_writer.add_scalar("Num epochs w/ no improvement: Self Supervised", num_epoch_no_improvement, epoch + 1)
            
        self._add_hparams_to_writer(task="ss")
        print("FINISHED TRAINING SS")

    def finetune_supervised(self):

        # TODO: DO NOT CONTINUE TO SUPERVISED IF SELF SUPERVISED STOPPED EARLY
        if self.config.optimizer_sup == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), self.config.lr_sup, momentum=0.9, weight_decay=0.0, nesterov=False)
        elif self.config.optimizer_sup == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), self.config.lr_sup, betas=(self.config.beta1_sup, self.config.beta2_sup), eps=self.config.eps_sup
            )

        if self.config.loss_function_sup == "binary_cross_entropy":
            criterion = nn.BCELoss()  # #model outputs sigmoid so no use of BCEwithLogits
        elif self.config.loss_function_sup == "dice":
            criterion = DiceLoss.dice_loss
        elif self.config.loss_function_sup == "mix_dice_bce":
            # https://discuss.pytorch.org/t/dice-loss-cross-entropy/53194
            raise NotImplementedError

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(self.config.patience_sup * 0.8), gamma=0.5)
        best_loss = 10000000000
        num_epoch_no_improvement = 0

        for epoch in range(self.config.nb_epoch_sup):

            self.stats.training_losses_sup = []
            self.stats.validation_losses_sup = []
            self.model.train()
            iteration = 0
            while True:  # go through all examples
                x, y = self.dataset.get_train(batch_size=self.config.batch_size_sup)
                if x is None:
                    break
                x, y = x.float().to(self.device), y.float().to(self.device)
                pred = self.model(x)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.tb_writer.add_scalar("Loss/train : Supervised", loss, (epoch + 1) * iteration)
                self.stats.training_losses_sup.append(loss.item())

                if (iteration + 1) % 5 == 0:
                    print(
                        "Epoch [{}/{}], iteration {}, Loss: {:.6f}".format(
                            epoch + 1, self.config.nb_epoch_sup, iteration + 1, np.average(self.stats.training_losses_sup)
                        )
                    )
                    sys.stdout.flush()
                iteration += 1

            with torch.no_grad():
                self.model.eval()
                x = 0
                while True:
                    x, y = self.dataset.get_val(batch_size=self.config.batch_size_sup)
                    if x is None:
                        break
                    x, y = x.float().to(self.device), y.float().to(self.device)
                    pred = self.model(x)
                    loss = criterion(pred, y)
                    self.tb_writer.add_scalar("Loss/Validation : Supervised", loss, (epoch + 1) * iteration)
                    self.stats.validation_losses_sup.append(loss.item())

            self.dataset.reset()
            scheduler.step(epoch)
            avg_training_loss_of_epoch = np.average(self.stats.training_losses_sup)
            self.tb_writer.add_scalar("Avg Loss Epoch/Training : Supervised", avg_training_loss_of_epoch, epoch + 1)
            avg_validation_loss_of_epoch = np.average(self.stats.validation_losses_sup)
            self.tb_writer.add_scalar("Avg Loss Epoch/Validation : Supervised", avg_validation_loss_of_epoch, epoch + 1)

            self.stats.avg_training_loss_per_epoch_sup.append(avg_training_loss_of_epoch)
            self.stats.avg_validation_loss_per_epoch_sup.append(avg_validation_loss_of_epoch)
            self.stats.iterations_sup.append(iteration)

            avg_training_loss_of_epoch = np.average(self.stats.training_losses_ss)
            avg_validation_loss_of_epoch = np.average(self.stats.validation_losses_ss)

            print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch + 1, avg_validation_loss_of_epoch, avg_training_loss_of_epoch))
            if avg_validation_loss_of_epoch < best_loss:
                print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, avg_validation_loss_of_epoch))
                best_loss = avg_validation_loss_of_epoch
                num_epoch_no_improvement = 0
                # save model
                torch.save(
                    {"epoch": epoch + 1, "state_dict": self.model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                    os.path.join(self.config.model_path_save, "weights.pt"),
                )
                print("Model Saved")
            else:
                print("Validation loss did not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss, num_epoch_no_improvement))
                num_epoch_no_improvement += 1
            if num_epoch_no_improvement >= self.config.patience_sup:
                print("Early Stopping SUP")
                self.stats.stopped_early_sup = True
                break
            sys.stdout.flush()
            self.tb_writer.add_scalar("Num epochs w/ no improvement: Supervised", num_epoch_no_improvement, epoch + 1)

        print("FINISHED TRAINING SUP")
        self._add_hparams_to_writer(task="sup")

    def _add_hparams_to_writer(self, task: str):

        # optimizer, loss , batch size
        if task == "ss" and self.config.self_supervised:
            self.tb_writer.add_hparams(
                {   
                    "initial_lr_ss": self.config.lr_ss,
                    "optimizer_ss": self.config.optimizer_ss,
                    "scheduler_ss": self.config.scheduler_ss,
                    "batch_size_ss": self.config.batch_size_ss,
                    "nr_epochs_ss": self.config.nb_epoch_ss
                },
                {
                    "final_train_loss_ss": self.stats.avg_training_loss_per_epoch_ss[-1],
                    "final_val_loss_ss": self.stats.avg_validation_loss_per_epoch_ss[-1],
                    "stopped_early_ss": self.stats.stopped_early_ss}
            )

        if task == "sup":
            self.tb_writer.add_hparams(
                {   
                    "cube_dimensions": self.dataset.cube_dimensions,
                    "initial_lr_sup": self.config.lr_sup,
                    "optimizer_sup": self.config.optimizer_sup,
                    "scheduler_sup": self.config.scheduler_sup,
                    "batch_size_sup": self.config.batch_size_sup,
                    "nr_epochs_sup": self.config.nb_epoch_sup
                },
                {
                    "final_train_loss_sup": self.stats.avg_training_loss_per_epoch_sup[-1],
                    "final_val_loss_sup": self.stats.avg_validation_loss_per_epoch_sup[-1],
                    "stopped_early_sup": self.stats.stopped_early_sup}
            )
    
    def close_writer(self):
        self.tb_writer.close()

    def _loadmodel(self, from_weights=True, from_latest_checkpoint=True):
        """
        Keyword Arguments:
            from_latest_checkpoint {bool} -- In case something happens that causes exiting and the experiment is to be resumed (default: {True})
        """

        self.model = UNet3D()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if from_weights:
            weight_dir = (
                self.config.model_path_save
                if from_latest_checkpoint and os.path.isfile(os.path.join(self.config.model_path_save, "weights.pt"))
                else self.config.weights
            )
            checkpoint = torch.load(weight_dir, map_location=self.device)
            state_dict = checkpoint["state_dict"]
            unParalled_state_dict = {}
            for key in state_dict.keys():
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            self.model.load_state_dict(unParalled_state_dict)  ### TODO: WHY IS THIS UNPARALLELED NECESSARY?

        self.model.to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(torch.cuda.device_count())])

    def get_stats(self):
        self.stats.get_statistics()


if __name__ == "__main__":
    pass
