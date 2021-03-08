import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import numpy as np
import time
from datetime import timedelta
from copy import deepcopy
from sklearn.metrics import precision_recall_fscore_support

from ACSConv.experiments.mylib.utils import categorical_to_one_hot

from unet_2d import UNet
from unet_3d import UNet3D as Unet3D_Counterpart_to_2D
from unet_2d import UnetACSWithClassifier
from unet_2d import UnetACSAxisAwareDecoder
from unet_2d import UnetACSWithClassifierOnly

from unet3d import UNet3D
from dataset import Dataset
from dataset_2d import Dataset2D
from dataset_pytorch import DatasetPytorch
from datasets_pytorch import DatasetsPytorch
from finetune_config import FineTuneConfig
from config import models_genesis_config
from stats import Statistics

from dice_loss import DiceLoss
from image_transformations import generate_pair
from utils import pad_if_necessary, save_object


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
        print("DEVICE: ", self.device)

    def finetune_self_supervised(self):

        self.start_time = time.time()

        if isinstance(self.dataset, list):

            copy1 = deepcopy(self.dataset)
            copy2 = deepcopy(self.dataset)
            for i in range(len(copy1)):
                copy1[i] = DatasetPytorch(copy1[i], self.config, type_="train", apply_mg_transforms=True)
                copy2[i] = DatasetPytorch(copy2[i], self.config, type_="val", apply_mg_transforms=True)

            train_dataset = DatasetsPytorch(
                datasets=copy1, type_="train", mode=self.config.mode, batch_size=self.config.batch_size_ss, apply_mg_transforms=True
            )
            train_data_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size_ss,
                num_workers=self.config.workers,
                collate_fn=DatasetsPytorch.custom_collate,
                pin_memory=True,
            )

            val_dataset = DatasetsPytorch(
                datasets=copy2, type_="val", mode=self.config.mode, batch_size=self.config.batch_size_ss, apply_mg_transforms=True
            )
            val_data_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size_ss,
                num_workers=self.config.workers,
                collate_fn=DatasetsPytorch.custom_collate,
                pin_memory=True,
            )

        else:

            train_dataset = DatasetPytorch(self.dataset, self.config, type_="train", apply_mg_transforms=True)
            train_data_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size_ss,
                num_workers=self.config.workers,
                collate_fn=DatasetPytorch.custom_collate,
                pin_memory=True,
            )

            val_dataset = DatasetPytorch(self.dataset, self.config, type_="val", apply_mg_transforms=True)
            val_data_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size_ss,
                num_workers=self.config.workers,
                collate_fn=DatasetPytorch.custom_collate,
                pin_memory=True,
            )

        num_train_samples = train_dataset.__len__()
        num_val_samples = val_dataset.__len__()
        print("{} TRAINING EXAMPLES".format(num_train_samples))
        print("{} VALIDATION EXAMPLES".format(num_val_samples))

        criterion = nn.MSELoss()
        criterion.to(self.device)
        if self.config.model.lower() == "unet_acs_with_cls":
            criterion_cls = nn.CrossEntropyLoss()
            criterion_cls.to(self.device)

        if self.epoch_ss_check > 0:
            print("RESUMING SS TRAINING FROM EPOCH {} out of max {}".format(self.epoch_ss_check, self.config.nb_epoch_ss))
            print(
                "PREVIOUS BEST SS LOSS: {} // NUM SS EPOCH WITH NO IMPROVEMENT: {}".format(
                    self.best_loss_ss, self.num_epoch_no_improvement_ss
                )
            )
            try:
                print("CURRNT SS LR: {}".format(self.scheduler_ss.get_last_lr()))
            except AttributeError:
                print("CURRNT SS LR: {}".format(self.optimizer_ss.param_groups[0]["lr"]))

        else:
            print("STARTING SS TRAINING FROM SCRATCH")

        for self.epoch_ss_current in range(self.epoch_ss_check, self.config.nb_epoch_ss):
            self.stats.training_losses_ss = []
            self.stats.validation_losses_ss = []
            if self.config.model.lower() == "unet_acs_with_cls":
                self.stats.training_losses_ss_cls = []
                self.stats.validation_losses_ss_cls = []
                self.stats.training_losses_ss_ae = []
                self.stats.validation_losses_ss_ae = []
            self.model.train()

            for iteration, (x_transform, y) in enumerate(train_data_loader):
                if iteration == 0 and ((self.epoch_ss_current) % 20 == 0):
                    start_time = time.time()

                x_transform, y = x_transform.float().to(self.device), y.float().to(self.device)
                if self.config.model.lower() in ("vnet_mg", "unet_3d", "unet_acs", "unet_acs_axis_aware_decoder", "unet_acs_with_cls"):
                    x_transform, y = pad_if_necessary(x_transform, y)

                pred = self.model(x_transform)

                if self.config.model.lower() == "unet_acs_with_cls":
                    pred_, x_cls, targets_cls = pred
                    x_cls, targets_cls = x_cls.to(self.device), targets_cls.to(self.device)
                    pred = pred_  # for reconstruction
                    loss_cls = criterion_cls(x_cls, targets_cls)
                    loss_cls.to(self.device)
                    loss_cls *= 0.1
                    self.tb_writer.add_scalar("Loss/train CLS: Self Supervised", loss_cls.item(), (self.epoch_ss_current + 1) * iteration)
                    self.stats.training_losses_ss_cls.append(loss_cls.item())
                    # print("CLS", loss_cls.item())
                    # if self.epoch_ss_current >= 20:

                loss = criterion(pred, y)
                loss.to(self.device)
                self.optimizer_ss.zero_grad()

                if self.config.model.lower() == "unet_acs_with_cls":
                    self.tb_writer.add_scalar("Loss/train AE : Self Supervised", loss.item(), (self.epoch_ss_current + 1) * iteration)
                    self.stats.training_losses_ss_ae.append(loss.item())
                    loss += loss_cls

                self.tb_writer.add_scalar("Loss/train : Self Supervised", loss.item(), (self.epoch_ss_current + 1) * iteration)
                loss.backward()
                self.optimizer_ss.step()
                self.stats.training_losses_ss.append(loss.item())

                if (iteration + 1) % int((int(train_dataset.__len__() / self.config.batch_size_ss)) / 5) == 0:

                    if self.config.model.lower() == "unet_acs_with_cls":
                        print(
                            "Epoch [{}/{}], iteration {}, TRAINING Loss: {:.6f}, TRAINING Loss(AE): {:.6f}, TRAINING Loss(CLS): {:.6f}".format(
                                self.epoch_ss_current + 1,
                                self.config.nb_epoch_ss,
                                iteration + 1,
                                np.average(self.stats.training_losses_ss),
                                np.average(self.stats.training_losses_ss_ae),
                                np.average(self.stats.training_losses_ss_cls),
                            )
                        )
                    else:
                        print(
                            "Epoch [{}/{}], iteration {}, TRAINING Loss: {:.6f}".format(
                                self.epoch_ss_current + 1, self.config.nb_epoch_ss, iteration + 1, np.average(self.stats.training_losses_ss)
                            )
                        )

                if iteration == 0 and ((self.epoch_ss_current) % 20 == 0):
                    timedelta_iter = timedelta(seconds=time.time() - start_time)
                    print("TIMEDELTA FOR ITERATION {}".format(str(timedelta_iter)))
                sys.stdout.flush()

            with torch.no_grad():
                self.model.eval()
                for iteration, (x_transform, y) in enumerate(val_data_loader):
                    if x_transform is None:
                        raise RuntimeError("THIS SHOULD NOT HAPPEN")

                    x_transform, y = x_transform.float().to(self.device), y.float().to(self.device)
                    if self.config.model.lower() in ("vnet_mg", "unet_3d", "unet_acs", "unet_acs_axis_aware_decoder", "unet_acs_with_cls"):
                        x_transform, y = pad_if_necessary(x_transform, y)
                    pred = self.model(x_transform)
                    if self.config.model.lower() == "unet_acs_with_cls":
                        pred_, x_cls, targets_cls = pred
                        x_cls, targets_cls = x_cls.to(self.device), targets_cls.to(self.device)
                        pred = pred_  # for reconstruction
                        loss_cls = criterion_cls(x_cls, targets_cls)
                        loss_cls *= 0.1
                        self.tb_writer.add_scalar(
                            "Loss/Validation CLS: Self Supervised", loss_cls.item(), (self.epoch_ss_current + 1) * iteration
                        )
                        self.stats.validation_losses_ss_cls.append(loss_cls.item())

                    loss = criterion(pred, y)
                    if self.config.model.lower() == "unet_acs_with_cls":
                        self.tb_writer.add_scalar(
                            "Loss/Validation AE : Self Supervised", loss.item(), (self.epoch_ss_current + 1) * iteration
                        )
                        self.stats.validation_losses_ss_ae.append(loss.item())
                        loss += loss_cls

                    self.tb_writer.add_scalar("Loss/Validation : Self Supervised", loss.item(), (self.epoch_ss_current + 1) * iteration)
                    self.stats.validation_losses_ss.append(loss.item())

            # a bunch of checks but think its pointless in an MP context
            train_dataset.reset()
            val_dataset.reset()

            if isinstance(self.dataset, list):
                for i in range(len(self.dataset)):
                    self.dataset[i].reset()
            else:
                self.dataset.reset()

            avg_training_loss_of_epoch = np.average(self.stats.training_losses_ss)
            self.tb_writer.add_scalar("Avg Loss Epoch/Training : Self Supervised", avg_training_loss_of_epoch, self.epoch_ss_current + 1)
            avg_validation_loss_of_epoch = np.average(self.stats.validation_losses_ss)
            self.tb_writer.add_scalar(
                "Avg Loss Epoch/Validation : Self Supervised", avg_validation_loss_of_epoch, self.epoch_ss_current + 1
            )

            if self.config.model.lower() == "unet_acs_with_cls":

                avg_training_loss_of_epoch_ae = np.average(self.stats.training_losses_ss_ae)
                avg_training_loss_of_epoch_cls = np.average(self.stats.training_losses_ss_cls)
                self.tb_writer.add_scalar(
                    "Avg Loss Epoch/Training AE : Self Supervised", avg_training_loss_of_epoch_ae, self.epoch_ss_current + 1
                )
                self.tb_writer.add_scalar(
                    "Avg Loss Epoch/Training CLS : Self Supervised", avg_training_loss_of_epoch_cls, self.epoch_ss_current + 1
                )

                avg_validation_loss_of_epoch_ae = np.average(self.stats.validation_losses_ss_ae)
                avg_validation_loss_of_epoch_cls = np.average(self.stats.validation_losses_ss_cls)
                self.tb_writer.add_scalar(
                    "Avg Loss Epoch/Validation AE : Self Supervised", avg_validation_loss_of_epoch_ae, self.epoch_ss_current + 1
                )
                self.tb_writer.add_scalar(
                    "Avg Loss Epoch/Validation CLS: Self Supervised", avg_validation_loss_of_epoch_cls, self.epoch_ss_current + 1
                )

            self.stats.avg_training_loss_per_epoch_ss.append(avg_training_loss_of_epoch)
            self.stats.avg_validation_loss_per_epoch_ss.append(avg_validation_loss_of_epoch)
            self.stats.iterations_ss.append(iteration)

            if self.config.scheduler_ss == "ReduceLROnPlateau":
                self.scheduler_ss.step(avg_validation_loss_of_epoch)
            else:
                self.scheduler_ss.step(self.epoch_ss_current)

            print("###### SELF SUPERVISED#######")
            if self.config.model.lower() == "unet_acs_with_cls":
                print(
                    "Epoch {}, Validation loss is {:.4f}, training loss is {:.4f}. \n Validation Loss AE is {}, Training loss AE is {:.4f} \n Validation Loss Cls is {}, Training loss cls is {:.4f}".format(
                        self.epoch_ss_current + 1,
                        avg_validation_loss_of_epoch,
                        avg_training_loss_of_epoch,
                        avg_validation_loss_of_epoch_ae,
                        avg_training_loss_of_epoch_ae,
                        avg_validation_loss_of_epoch_cls,
                        avg_training_loss_of_epoch_cls,
                    )
                )
            else:
                print(
                    "Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(
                        self.epoch_ss_current + 1, avg_validation_loss_of_epoch, avg_training_loss_of_epoch
                    )
                )

            try:
                print("CURRENT SS LR: {}, SCHEDULER: {}".format(self.scheduler_ss.get_last_lr(), self.config.scheduler_ss))
            except AttributeError:
                print("CURRENT SS LR: {}, SCHEDULER: {}".format(self.optimizer_ss.param_groups[0]["lr"], self.config.scheduler_ss))

            if avg_validation_loss_of_epoch < self.best_loss_ss:
                print("Validation loss decreased from {:.4f} to {:.4f}".format(self.best_loss_ss, avg_validation_loss_of_epoch))
                self.best_loss_ss = avg_validation_loss_of_epoch
                self.num_epoch_no_improvement_ss = 0
                self._save_model("ss")
            else:
                print(
                    "Validation loss did not decrease from {:.4f}, num_epoch_no_improvement {}".format(
                        self.best_loss_ss, self.num_epoch_no_improvement_ss + 1
                    )
                )
                self.num_epoch_no_improvement_ss += 1
                self._save_model("ss", suffix="_no_decrease")
                # self._save_num_epochs_no_improvement("ss")
            if self.num_epoch_no_improvement_ss >= self.config.patience_ss_terminate:
                print("Early Stopping SS")
                self.stats.stopped_early_ss = True
                break
            sys.stdout.flush()
            self.tb_writer.add_scalar(
                "Num epochs w/ no improvement: Self Supervised", self.num_epoch_no_improvement_ss, self.epoch_ss_current + 1
            )

        self.ss_timedelta = timedelta(seconds=time.time() - self.start_time)
        self._add_completed_flag_to_last_checkpoint_saved(phase="ss")
        print("FINISHED TRAINING SS")

    def finetune_supervised(self):

        self.start_time = time.time()

        if isinstance(self.dataset, list):

            copy1 = deepcopy(self.dataset)
            copy2 = deepcopy(self.dataset)

            for i in range(len(copy1)):
                copy1[i] = DatasetPytorch(copy1[i], self.config, type_="train", apply_mg_transforms=False)
                copy2[i] = DatasetPytorch(copy2[i], self.config, type_="val", apply_mg_transforms=False)

            train_dataset = DatasetsPytorch(
                datasets=copy1, type_="train", mode=self.config.mode, batch_size=self.config.batch_size_sup, apply_mg_transforms=False
            )
            train_data_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size_sup,
                num_workers=self.config.workers,
                collate_fn=DatasetsPytorch.custom_collate,
                pin_memory=True,
            )

            val_dataset = DatasetsPytorch(
                datasets=copy2, type_="val", mode=self.config.mode, batch_size=self.config.batch_size_sup, apply_mg_transforms=False
            )
            val_data_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size_sup,
                num_workers=self.config.workers,
                collate_fn=DatasetsPytorch.custom_collate,
                pin_memory=True,
            )

        else:

            train_dataset = DatasetPytorch(self.dataset, self.config, type_="train", apply_mg_transforms=False)
            train_data_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size_sup,
                num_workers=self.config.workers,
                collate_fn=DatasetPytorch.custom_collate,
                pin_memory=True,
            )

            val_dataset = DatasetPytorch(self.dataset, self.config, type_="val", apply_mg_transforms=False)
            val_data_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size_sup,
                num_workers=self.config.workers,
                collate_fn=DatasetPytorch.custom_collate,
                pin_memory=True,
            )

        if self.config.loss_function_sup.lower() == "binary_cross_entropy":
            criterion = nn.BCELoss()  # #model outputs sigmoid so no use of BCEwithLogits
            criterion.to(self.device)
        elif self.config.loss_function_sup.lower() == "dice":
            criterion = DiceLoss.dice_loss
        elif self.config.loss_function_sup.lower() == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        elif self.config.loss_function_sup == "mix_dice_bce":
            # https://discuss.pytorch.org/t/dice-loss-cross-entropy/53194
            raise NotImplementedError

        if self.epoch_sup_check > 0:
            print("RESUMING SUP TRAINING FROM EPOCH {} out of max {}".format(self.epoch_sup_check, self.config.nb_epoch_sup))
            print(
                "PREVIOUS BEST SUP LOSS: {} // NUM SUP EPOCHS WITH NO IMPROVEMENT {}".format(
                    self.best_loss_sup, self.num_epoch_no_improvement_sup
                )
            )
            try:
                print("CURRNT SUP LR: {}, SCHEDULER: {}".format(self.scheduler_sup.get_last_lr(), self.config.scheduler_sup))
            except AttributeError:
                print("CURRNT SUP LR: {}, SCHEDULER: {}".format(self.optimizer_sup.param_groups[0]["lr"], self.config.scheduler_sup))
        else:
            print("STARTING SUP TRAINING FROM SCRATCH")

        print("{} TRAINING EXAMPLES".format(train_dataset.__len__()))
        print("{} VALIDATION EXAMPLES".format(val_dataset.__len__()))

        for self.epoch_sup_current in range(self.epoch_sup_check, self.config.nb_epoch_sup):

            self.stats.training_losses_sup = []
            self.stats.validation_losses_sup = []
            self.model.train()

            if "unet_acs_cls_only" in self.config.model.lower():
                self.stats.training_recall = []
                self.stats.training_precision = []
                self.stats.training_f1 = []
                self.stats.testing_recall = []
                self.stats.testing_precision = []
                self.stats.testing_f1 = []
                self.stats.training_accuracy = []
                self.stats.testing_accuracy = []

            for iteration, (x, y) in enumerate(train_data_loader):

                if iteration == 0 and ((self.epoch_sup_current + 1) % 20 == 0):
                    start_time = time.time()

                if x is None:
                    raise RuntimeError

                x, y = x.float().to(self.device), y.float().to(self.device)
                if self.config.model.lower() in (
                    "vnet_mg",
                    "unet_3d",
                    "unet_acs",
                    "unet_acs_axis_aware_decoder",
                    "unet_acs_with_cls",
                    "unet_acs_cls_only",
                ):
                    x, y = pad_if_necessary(x, y)
                if "fcn_resnet18" in self.config.model.lower():
                    # expects 3 channel input
                    x = torch.cat((x, x, x), dim=1)
                    # 2 channel output of network
                    y = categorical_to_one_hot(y, dim=1, expand_dim=False)

                pred = self.model(x)

                if "unet_acs_cls_only" in self.config.model.lower():
                    x_cls, y = pred
                    x_cls, y = x_cls.to(self.device), y.to(self.device)
                    pred = x_cls
                    pr, rc, f1, accu = self.compute_precision_recall_f1(pred.cpu(), y.cpu())
                    self.stats.training_precision.append(pr)
                    self.stats.training_recall.append(rc)
                    self.stats.training_f1.append(f1)
                    self.stats.training_accuracy.append(accu)

                loss = criterion(pred, y)
                loss.to(self.device)
                self.optimizer_sup.zero_grad()
                loss.backward()
                self.optimizer_sup.step()
                self.tb_writer.add_scalar("Loss/train : Supervised", loss.item(), (self.epoch_sup_current + 1) * iteration)
                self.stats.training_losses_sup.append(loss.item())

                if (iteration + 1) % int((int(train_dataset.__len__() / self.config.batch_size_sup)) / 5) == 0:
                    print(
                        "Epoch [{}/{}], iteration {}, Loss: {:.6f}".format(
                            self.epoch_sup_current + 1, self.config.nb_epoch_sup, iteration + 1, np.average(self.stats.training_losses_sup)
                        )
                    )
                if iteration == 0 and ((self.epoch_sup_current + 1) % 20 == 0):
                    timedelta_iter = timedelta(seconds=time.time() - start_time)
                    print("TIMEDELTA FOR ITERATION {}".format(str(timedelta_iter)))
                sys.stdout.flush()

            with torch.no_grad():
                self.model.eval()
                for iteration, (x, y) in enumerate(val_data_loader):
                    # x, y = self.dataset.get_val(batch_size=self.config.batch_size_sup)
                    # if x is None:
                    #    break
                    x, y = x.float().to(self.device), y.float().to(self.device)
                    if self.config.model.lower() in (
                        "vnet_mg",
                        "unet_3d",
                        "unet_acs",
                        "unet_acs_axis_aware_decoder",
                        "unet_acs_with_cls",
                        "unet_acs_cls_only",
                    ):
                        x, y = pad_if_necessary(x, y)
                    if "fcn_resnet18" in self.config.model.lower():
                        # expects 3 channel input
                        x = torch.cat((x, x, x), dim=1)
                        # 2 channel output of network
                        y = categorical_to_one_hot(y, dim=1, expand_dim=False)

                    pred = self.model(x)

                    if "unet_acs_cls_only" in self.config.model.lower():
                        x_cls, y = pred  # y is target of cls
                        x_cls, y = x_cls.to(self.device), y.to(self.device)
                        pred = x_cls  # rename pred to not add if statements to loss calculation
                        pr, rc, f1, accu = self.compute_precision_recall_f1(pred.cpu(), y.cpu())
                        self.stats.testing_precision.append(pr)
                        self.stats.testing_recall.append(rc)
                        self.stats.testing_f1.append(f1)
                        self.stats.testing_accuracy.append(accu)

                    loss = criterion(pred, y)
                    self.tb_writer.add_scalar("Loss/Validation : Supervised", loss.item(), (self.epoch_sup_current + 1) * iteration)
                    self.stats.validation_losses_sup.append(loss.item())

            # a bunch of checks but think its pointless in an MP context there is an automatic mechanims in place
            train_dataset.reset()
            val_dataset.reset()

            if isinstance(self.dataset, list):
                for i in range(len(self.dataset)):
                    self.dataset[i].reset()
            else:
                self.dataset.reset()

            if "unet_acs_cls_only" in self.config.model.lower():

                avg_training_precision_of_epoch = np.average(self.stats.training_precision)
                self.tb_writer.add_scalar("Avg Precision of Epoch (Training)", avg_training_precision_of_epoch, self.epoch_sup_current + 1)
                avg_training_recall_of_epoch = np.average(self.stats.training_recall)
                self.tb_writer.add_scalar("Avg Recall of Epoch (Training)", avg_training_recall_of_epoch, self.epoch_sup_current + 1)
                avg_training_f1_of_epoch = np.average(self.stats.training_f1)
                self.tb_writer.add_scalar("Avg F1 of Epoch (Training)", avg_training_f1_of_epoch, self.epoch_sup_current + 1)
                avg_training_accuracy_of_epoch = np.average(self.stats.training_accuracy)
                self.tb_writer.add_scalar("Avg Accuracy of Epoch(Training)", avg_training_accuracy_of_epoch, self.epoch_sup_current + 1)

                avg_testing_precision_of_epoch = np.average(self.stats.testing_precision)
                self.tb_writer.add_scalar("Avg Precision of Epoch (Testing)", avg_testing_precision_of_epoch, self.epoch_sup_current + 1)
                avg_testing_recall_of_epoch = np.average(self.stats.testing_recall)
                self.tb_writer.add_scalar("Avg Recall of Epoch (Testing)", avg_testing_recall_of_epoch, self.epoch_sup_current + 1)
                avg_testing_f1_of_epoch = np.average(self.stats.testing_f1)
                self.tb_writer.add_scalar("Avg F1 of Epoch (Testing)", avg_testing_f1_of_epoch, self.epoch_sup_current + 1)
                avg_testing_accuracy_of_epoch = np.average(self.stats.testing_accuracy)
                self.tb_writer.add_scalar("Avg Accuracy of Epoch(Testing)", avg_testing_accuracy_of_epoch, self.epoch_sup_current + 1)

            avg_training_loss_of_epoch = np.average(self.stats.training_losses_sup)
            self.tb_writer.add_scalar("Avg Loss Epoch/Training : Supervised", avg_training_loss_of_epoch, self.epoch_sup_current + 1)
            avg_validation_loss_of_epoch = np.average(self.stats.validation_losses_sup)
            self.tb_writer.add_scalar("Avg Loss Epoch/Validation : Supervised", avg_validation_loss_of_epoch, self.epoch_sup_current + 1)

            self.stats.avg_training_loss_per_epoch_sup.append(avg_training_loss_of_epoch)
            self.stats.avg_validation_loss_per_epoch_sup.append(avg_validation_loss_of_epoch)
            self.stats.iterations_sup.append(iteration)

            if self.config.scheduler_sup.lower() == "reducelronplateau":
                self.scheduler_sup.step(avg_validation_loss_of_epoch)
            else:
                self.scheduler_sup.step(self.epoch_sup_current)

            if "unet_acs_cls_only" in self.config.model.lower():
                print(
                    "Epoch {}, validation loss is {:.4f}, training loss is {:.4f} \n Precision:{:.4f} / {:.4f} ; Recall:{:.4f} / {:.4f} ; F1:{:.4f} / {:.4f}, Accuracy:{:.4f} / {:.4f} ".format(
                        self.epoch_sup_current + 1,
                        avg_validation_loss_of_epoch,
                        avg_training_loss_of_epoch,
                        avg_training_precision_of_epoch,
                        avg_testing_precision_of_epoch,
                        avg_training_recall_of_epoch,
                        avg_testing_recall_of_epoch,
                        avg_training_f1_of_epoch,
                        avg_testing_f1_of_epoch,
                        avg_training_accuracy_of_epoch,
                        avg_testing_accuracy_of_epoch,
                    )
                )
            else:
                print(
                    "Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(
                        self.epoch_sup_current + 1, avg_validation_loss_of_epoch, avg_training_loss_of_epoch
                    )
                )

            try:
                print("CURRNT SUP LR: {}".format(self.scheduler_sup.get_last_lr()))
            except AttributeError:
                print("CURRENT SUP LR: {}".format(self.optimizer_sup.param_groups[0]["lr"]))

            if avg_validation_loss_of_epoch < self.best_loss_sup:
                print("Validation loss decreased from {:.4f} to {:.4f}".format(self.best_loss_sup, avg_validation_loss_of_epoch))
                self.best_loss_sup = avg_validation_loss_of_epoch
                self.num_epoch_no_improvement_sup = 0
                self._save_model("sup")
            else:
                print(
                    "Validation loss did not decrease from {:.4f}, num_epoch_no_improvement {}".format(
                        self.best_loss_sup, self.num_epoch_no_improvement_sup + 1
                    )
                )
                self.num_epoch_no_improvement_sup += 1
                self._save_model("sup", suffix="_no_decrease")
                # self._save_num_epochs_no_improvement("sup")
            if self.num_epoch_no_improvement_sup >= self.config.patience_sup_terminate:
                print("Early Stopping SUP")
                self.stats.stopped_early_sup = True
                break
            sys.stdout.flush()
            self.tb_writer.add_scalar(
                "Num epochs w/ no improvement: Supervised", self.num_epoch_no_improvement_sup, self.epoch_sup_current + 1
            )

        self.sup_timedelta = timedelta(seconds=time.time() - self.start_time)
        self._add_completed_flag_to_last_checkpoint_saved(phase="sup")
        print("FINISHED TRAINING SUP")

    @staticmethod
    def compute_precision_recall_f1(prediction, target):
        # make target binary
        predicted_labels = prediction.argmax(1)
        # multi class f1-score will be done macro averaging as all classes are equally important #alternatily (emphasize z axis??)
        pr, rec, f1, _ = precision_recall_fscore_support(target, predicted_labels, average="macro")
        accuracy = float((predicted_labels == target).sum()) / float(len(target))
        return pr, rec, f1, accuracy

    def add_hparams_to_writer(self):

        hpa_dict = {
            "cube_dimensions": self.dataset.cube_dimensions
            if isinstance(self.dataset, Dataset) or isinstance(self.dataset, Dataset2D)
            else self.dataset[0].cube_dimensions,
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
            "nr_epochs_sup": self.config.nb_epoch_sup,
        }

        self.ss_timedelta = 0 if (not hasattr(self, "ss_timedelta")) else (self.ss_timedelta.seconds // 60 % 60)  # conversion to minutes
        self.sup_timedelta = 0 if (not hasattr(self, "sup_timedelta")) else (self.sup_timedelta.seconds // 60 % 60)
        self.stats.last_avg_training_loss_per_epoch_ss = (
            0 if not self.stats.avg_training_loss_per_epoch_ss else self.stats.avg_training_loss_per_epoch_ss[-1]
        )
        self.stats.last_avg_validation_loss_per_epoch_ss = (
            0 if not self.stats.avg_validation_loss_per_epoch_ss else self.stats.avg_validation_loss_per_epoch_ss[-1]
        )
        self.stats.last_avg_training_loss_per_epoch_sup = (
            0 if not self.stats.avg_training_loss_per_epoch_sup else self.stats.avg_training_loss_per_epoch_sup[-1]
        )
        self.stats.last_avg_validation_loss_per_epoch_sup = (
            0 if not self.stats.avg_validation_loss_per_epoch_sup else self.stats.avg_validation_loss_per_epoch_sup[-1]
        )

        met_dict = {
            "final_train_loss_ss": self.stats.last_avg_training_loss_per_epoch_ss,
            "final_val_loss_ss": self.stats.last_avg_validation_loss_per_epoch_ss,
            "stopped_early_ss": self.stats.stopped_early_ss,
            "training_time_ss": self.ss_timedelta,
            "final_train_loss_sup": self.stats.last_avg_training_loss_per_epoch_sup,
            "final_val_loss_sup": self.stats.last_avg_validation_loss_per_epoch_sup,
            "stopped_early_sup": self.stats.stopped_early_sup,
            "training_time_sup": self.sup_timedelta,
        }

        # if hasattr(self, "tester"):
        #    if isinstance(self.tester.dice, float):
        #        met_dict.update({"test_dice": self.tester.dice, "test_jaccard": self.tester.jaccard})
        #    elif isinstance(self.tester.dice, list):
        #        met_dict.update({"test_dice_{}".format(str(i)): dice for i, dice in enumerate(self.tester.dice)})
        #        met_dict.update({"test_jaccard_{}".format(str(i)): jacd for i, jacd in enumerate(self.tester.jaccard)})

        self.tb_writer.add_hparams(hparam_dict=hpa_dict, metric_dict=met_dict)
        self.tb_writer.flush()
        self.tb_writer.close()

    def load_model(self, **kwargs):

        from ACSConv.acsconv.converters import ACSConverter
        from ACSConv.experiments.lidc.resnet import FCNResNet

        acs_kernel_split = kwargs.get("acs_kernel_split", None)
        if "unet_acs" not in self.config.model.lower():
            assert acs_kernel_split is None

        if self.config.model.lower() == "vnet_mg":
            print("Loading VNET_MG")
            self.model = UNet3D()

        elif self.config.model.lower() == "unet_2d":
            print("LOADING UNET 2D")
            self.model = UNet(n_channels=1, n_classes=1, bilinear=True, apply_sigmoid_to_output=True)

        elif self.config.model.lower() == "unet_acs":
            print("LOADING UNET_2D_ACS")
            self.model = UNet(n_channels=1, n_classes=1, bilinear=True, apply_sigmoid_to_output=True)
            self.model = ACSConverter(self.model, acs_kernel_split=acs_kernel_split)

        elif self.config.model.lower() == "unet_acs_with_cls":
            print("LOADING UNET_ACS_CLS")
            self.model = UnetACSWithClassifier(n_channels=1, n_classes=1, bilinear=True, apply_sigmoid_to_output=True)
            self.model = ACSConverter(self.model, acs_kernel_split=acs_kernel_split)

        elif self.config.model.lower() == "unet_acs_cls_only":
            print("LOADING UNET_ACS_CLS_ONLY")
            self.model = UnetACSWithClassifierOnly(n_channels=1, n_classes=1, bilinear=True, apply_sigmoid_to_output=True)
            self.model = ACSConverter(self.model, acs_kernel_split=acs_kernel_split)

        elif self.config.model.lower() == "unet_acs_cls_only_frozen_encoder":
            print("LOADING UNET_ACS_CLS_ONLY_FROZEN_ENCODER")
            self.model = UnetACSWithClassifierOnly(
                n_channels=1, n_classes=1, bilinear=True, apply_sigmoid_to_output=True, freeze_encoder=True
            )
            self.model = ACSConverter(self.model, acs_kernel_split=acs_kernel_split)

        elif self.config.model.lower() == "unet_acs_axis_aware_decoder":
            print("LOADING UNET_ACS_AXIS_AWARE_DECODER")
            self.model = UnetACSAxisAwareDecoder(n_channels=1, n_classes=1, bilinear=True, apply_sigmoid_to_output=True)
            self.model = ACSConverter(self.model, acs_kernel_split=acs_kernel_split)

        elif self.config.model.lower() == "unet_3d":
            print("LOADING UNET 3D")
            self.model = Unet3D_Counterpart_to_2D(n_channels=1, n_classes=1, bilinear=True, apply_sigmoid_to_output=True)

        elif self.config.model.lower() == "fcn_restnet18":
            print("LOADING FCN_RESNET_18")
            self.model = FCNResNet(pretrained=False, num_classes=2)

        elif self.config.model.lower() == "fcn_resnet18_acs":
            print("LOADING FCN_RESNET_18_ACS")
            self.model = FCNResNet(pretrained=False, num_classes=2)
            self.model = ACSConverter(self.model, acs_kernel_split=acs_kernel_split)

        elif self.config.model.lower() == "fcn_resnet18_acs_pretrained_imgnet":
            print("LOADING FCN_RESNET_18_pretrained_img_net")
            self.model = FCNResNet(pretrained=True, num_classes=2)
            self.model = ACSConverter(self.model, acs_kernel_split=acs_kernel_split)

        self.model.to(self.device)

        from_latest_checkpoint = kwargs.get("from_latest_checkpoint", False)
        from_latest_improvement_ss = kwargs.get("from_latest_improvement_ss", False)
        from_provided_weights = kwargs.get("from_provided_weights", False)
        from_scratch = kwargs.get("from_scratch", False)
        from_directory = kwargs.get("from_directory", False)
        from_path = kwargs.get("from_path", False)
        ensure_sup_is_completed = kwargs.get("ensure_sup_is_completed", False)

        if from_directory is not False:
            specific_weight_dir = kwargs.get("directory", None)
            assert specific_weight_dir is not None, "Specifiy weight dir to load"

        if from_path is not False:
            specific_weight_path = kwargs.get("path", None)
            phase = kwargs.get("phase", None)
            assert phase in ("ss", "sup")
            assert specific_weight_path is not None, "Specifiy weight path to load"
            if ("FROM_PROVIDED_WEIGHTS_SUP_ONLY_lidc_VNET_MG" in specific_weight_path) or (
                "FROM_PROVIDED_WEIGHTS_SS_AND_SUP_lidc_VNET_MG" in specific_weight_path
            ):
                specific_weight_path_split = specific_weight_path.split("/")
                specific_weight_path_split[1] = "FROM_PROVIDED_WEIGHTS_lidc_VNET_MG"
                specific_weight_path = "/".join(specific_weight_path_split)

            assert os.path.isfile(specific_weight_path), "{} is NOT a file".format(specific_weight_path)

        DEFAULTING_TO_LAST_SS = False

        if from_latest_checkpoint:

            # for doing ss and sup finetuning
            if self.config.resume_ss and self.config.resume_sup:
                completed_ss = self.ss_has_been_completed()

                if completed_ss is False:
                    weight_dir_no_decrease = (
                        os.path.join(self.config.model_path_save, "weights_ss_no_decrease.pt")
                        if os.path.isfile(os.path.join(self.config.model_path_save, "weights_ss_no_decrease.pt"))
                        else None
                    )
                    weight_dir = (
                        os.path.join(self.config.model_path_save, "weights_ss.pt")
                        if os.path.isfile(os.path.join(self.config.model_path_save, "weights_ss.pt"))
                        else None
                    )
                    if weight_dir_no_decrease is not None:
                        weight_dir = self._get_dir_with_more_advanced_epoch(weight_dir, weight_dir_no_decrease, phase="ss")
                    self._loadparams(dir=weight_dir, phase="ss")

                else:
                    weight_dir_no_decrease = (
                        os.path.join(self.config.model_path_save, "weights_sup_no_decrease.pt")
                        if os.path.isfile(os.path.join(self.config.model_path_save, "weights_sup_no_decrease.pt"))
                        else None
                    )
                    weight_dir = (
                        os.path.join(self.config.model_path_save, "weights_sup.pt")
                        if os.path.isfile(os.path.join(self.config.model_path_save, "weights_sup.pt"))
                        else None
                    )
                    if weight_dir_no_decrease is not None and weight_dir is not None:
                        weight_dir = self._get_dir_with_more_advanced_epoch(weight_dir, weight_dir_no_decrease, phase="sup")
                        self._loadparams(dir=weight_dir, phase="sup")

                    elif weight_dir is None:
                        # if it didnt have time to do a full sup epoch
                        DEFAULTING_TO_LAST_SS = True
                        print("IT SEEMS SS COMPLETED BUT NO SUP EPOCH WAS COMPLETED, RESUMING FROM LATEST SS CHECKPOINT")
                        weight_dir = (
                            os.path.join(self.config.model_path_save, "weights_ss.pt")
                            if os.path.isfile(os.path.join(self.config.model_path_save, "weights_ss.pt"))
                            else None
                        )
                        if weight_dir is None:
                            raise FileNotFoundError("Could not find weights to load")
                        self._loadparams(fresh_params=True, phase="sup")

            elif self.config.resume_ss:
                # weight_dir will always exist from the 1st epoch
                weight_dir_no_decrease = (
                    os.path.join(self.config.model_path_save, "weights_ss_no_decrease.pt")
                    if os.path.isfile(os.path.join(self.config.model_path_save, "weights_ss_no_decrease.pt"))
                    else None
                )
                weight_dir = (
                    os.path.join(self.config.model_path_save, "weights_ss.pt")
                    if os.path.isfile(os.path.join(self.config.model_path_save, "weights_ss.pt"))
                    else None
                )
                if weight_dir_no_decrease is not None:
                    weight_dir = self._get_dir_with_more_advanced_epoch(weight_dir, weight_dir_no_decrease, phase="ss")
                if weight_dir is None:
                    raise FileNotFoundError("Could not find provided weights to load")
                self._loadparams(dir=weight_dir, phase="ss")

            elif self.config.resume_sup:
                weight_dir_no_decrease = (
                    os.path.join(self.config.model_path_save, "weights_sup_no_decrease.pt")
                    if os.path.isfile(os.path.join(self.config.model_path_save, "weights_sup_no_decrease.pt"))
                    else None
                )
                weight_dir = (
                    os.path.join(self.config.model_path_save, "weights_sup.pt")
                    if os.path.isfile(os.path.join(self.config.model_path_save, "weights_sup.pt"))
                    else None
                )
                if weight_dir_no_decrease is not None:
                    weight_dir = self._get_dir_with_more_advanced_epoch(weight_dir, weight_dir_no_decrease, phase="sup")
                if weight_dir is None:
                    raise FileNotFoundError("Could not find provided weights to load")
                self._loadparams(dir=weight_dir, phase="sup")

        if from_latest_improvement_ss:  # Transition
            weight_dir = (
                os.path.join(self.config.model_path_save, "weights_ss.pt")
                if os.path.isfile(os.path.join(self.config.model_path_save, "weights_ss.pt"))
                else None
            )
            if weight_dir is None:
                raise FileNotFoundError("Could not find latest SS Improvement checkpoint to start from")
            self._loadparams(fresh_params=True, phase="sup")

        if from_provided_weights:
            weight_dir = self.config.weights if os.path.isfile(self.config.weights) else None
            if weight_dir is None:
                raise FileNotFoundError("Could not find provided weights to load")
            self._loadparams(fresh_params=True, phase="both")

        if from_directory:
            # assuming for now will only load model after doing ss
            weight_dir = os.path.join(specific_weight_dir, "weights_ss.pt")
            self._loadparams(fresh_params=True, phase="both")

        if from_path:
            # path specifies the actual file, eg: weights_sup.pt
            weight_dir = specific_weight_path
            self._loadparams(fresh_params=True, phase="both")

        if from_scratch:
            weight_dir = None
            self._loadparams(fresh_params=True, phase="both")

        if weight_dir is None:
            print("Loading Model with random weights")

        else:
            print("LOADING WEIGHTS FROM {}".format(weight_dir))
            checkpoint = torch.load(weight_dir, map_location=self.device)
            if self.config.resume_ss and self.config.resume_sup:
                completed_ss = self.ss_has_been_completed()
                if not completed_ss:
                    state_dict = checkpoint["model_state_dict_ss"]
                else:  # could be shortened but it's nice to see the logic
                    if DEFAULTING_TO_LAST_SS:
                        state_dict = checkpoint["model_state_dict_ss"]
                    else:
                        state_dict = checkpoint["model_state_dict_sup"]

            if self.config.resume_ss or from_latest_improvement_ss:
                print("Loaded Model State dict from model_state_dict_ss checkpoint key")
                state_dict = checkpoint["model_state_dict_ss"]
            if self.config.resume_sup:
                print("Loaded Model State dict from model_state_dict_sup checkpoint key")
                state_dict = checkpoint["model_state_dict_sup"]
            if from_provided_weights:
                print("Loaded Model State dict from state_dict checkpoint key")
                state_dict = checkpoint["state_dict"]
            if from_directory:
                print("Loaded Model State dict from model_state_dict_ss checkpoint key")
                state_dict = checkpoint["model_state_dict_ss"]
            if from_path:
                if phase == "sup":
                    print("Loaded Model State dict from model_state_dict_sup checkpoint key")
                    state_dict = checkpoint["model_state_dict_sup"]
                elif phase == "ss":
                    print("Loaded Model State dict from model_state_dict_ss checkpoint key")
                    state_dict = checkpoint["model_state_dict_ss"]
            if ensure_sup_is_completed:
                assert checkpoint["completed_sup"] is True, "Supervison was not completed"
            # assert checkpoint["completed_sup"] is True, "Supervison was not completed, {}".format(kwargs)

            unParalled_state_dict = {}
            for key in state_dict.keys():
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            try:
                res = self.model.load_state_dict(unParalled_state_dict)
            except RuntimeError as e:
                if "Missing key(s) in state_dict" in str(e):
                    unParalled_state_dict = self._convert_state_dict_between_archs(unParalled_state_dict)
                    res = self.model.load_state_dict(unParalled_state_dict, strict=False)
            print("LOAD STATE DICT OUTPUT:", res)
        convert_acs = kwargs.get("convert_acs", False)
        if convert_acs is True:
            # can only convert from 2D to ACS
            assert isinstance(self.model, UNet)
            self.model = ACSConverter(self.model, acs_kernel_split=acs_kernel_split)
            print("CONVERTED MODEL FROM UNET 2D WITH ACS CONVERTER")
            # override config for from now on to use ACS MODEL
            if "unet" in self.config.model.lower():
                self.config.model = "UNET_ACS"
            elif "resnet18" in self.config.mode.lower():
                self.config.model = "FCN_RESNET18_ACS"
            save_object(self.config, "config", self.config.object_dir)

        nr_devices = len([i for i in range(torch.cuda.device_count())])
        print("FOUND {} CUDA DEVICES".format(nr_devices))
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(torch.cuda.device_count())])

    def _add_completed_flag_to_last_checkpoint_saved(self, phase: str):

        if phase == "ss":
            weight_dir = (
                os.path.join(self.config.model_path_save, "weights_ss.pt")
                if os.path.isfile(os.path.join(self.config.model_path_save, "weights_ss.pt"))
                else None
            )
            checkpoint = torch.load(weight_dir, map_location=self.device)
            checkpoint["completed_ss"] = True
            torch.save(checkpoint, os.path.join(self.config.model_path_save, "weights_ss.pt"))

            weight_dir_no_decrease = (
                os.path.join(self.config.model_path_save, "weights_ss_no_decrease.pt")
                if os.path.isfile(os.path.join(self.config.model_path_save, "weights_ss_no_decrease.pt"))
                else None
            )
            if weight_dir_no_decrease is not None:
                checkpoint = torch.load(weight_dir_no_decrease, map_location=self.device)
                checkpoint["completed_ss"] = True
                torch.save(checkpoint, os.path.join(self.config.model_path_save, "weights_ss_no_decrease.pt"))

            print("ADDED COMPLETED SS FLAG TO CHECKPOINT")

        elif phase == "sup":
            weight_dir = (
                os.path.join(self.config.model_path_save, "weights_sup.pt")
                if os.path.isfile(os.path.join(self.config.model_path_save, "weights_sup.pt"))
                else None
            )
            checkpoint = torch.load(weight_dir, map_location=self.device)
            checkpoint["completed_sup"] = True
            torch.save(checkpoint, os.path.join(self.config.model_path_save, "weights_sup.pt"))

            weight_dir_no_decrease = (
                os.path.join(self.config.model_path_save, "weights_sup_no_decrease.pt")
                if os.path.isfile(os.path.join(self.config.model_path_save, "weights_sup_no_decrease.pt"))
                else None
            )
            if weight_dir_no_decrease is not None:
                checkpoint = torch.load(weight_dir_no_decrease, map_location=self.device)
                checkpoint["completed_sup"] = True
                torch.save(checkpoint, os.path.join(self.config.model_path_save, "weights_sup_no_decrease.pt"))
            print("ADDED COMPLETED SUP FLAG TO CHECKPOINT")

        else:
            raise ValueError("Invalid phase")

    def ss_has_been_completed(self):

        weight_dir_ss = (
            os.path.join(self.config.model_path_save, "weights_ss.pt")
            if os.path.isfile(os.path.join(self.config.model_path_save, "weights_ss.pt"))
            else None
        )
        checkpoint = torch.load(weight_dir_ss, map_location=self.device)
        return checkpoint.get("completed_ss", False)

    def sup_has_been_completed(self):

        weight_dir_sup = (
            os.path.join(self.config.model_path_save, "weights_sup.pt")
            if os.path.isfile(os.path.join(self.config.model_path_save, "weights_ss.pt"))
            else None
        )
        checkpoint = torch.load(weight_dir_sup, map_location=self.device)
        return checkpoint.get("completed_sup", False)

    def _save_model(self, phase: str, suffix=""):
        """
        Args:
            phase (str): ["ss" or "sup"]
        """

        assert phase == "ss" or phase == "sup"

        if phase == "ss":
            torch.save(
                {
                    "epoch_ss": self.epoch_ss_current + 1,
                    "model_state_dict_ss": self.model.state_dict()
                    if "acs" not in self.config.model.lower()
                    else self.model.module.state_dict(),
                    "optimizer_state_dict_ss": self.optimizer_ss.state_dict(),
                    "scheduler_state_dict_ss": self.scheduler_ss.state_dict(),
                    "num_epoch_no_improvement_ss": self.num_epoch_no_improvement_ss,
                    "best_loss_ss": self.best_loss_ss,
                },
                os.path.join(self.config.model_path_save, "weights_ss{}.pt".format(suffix)),
            )
            print("Model Saved in {} \n".format(os.path.join(self.config.model_path_save, "weights_ss{}.pt".format(suffix))))

        elif phase == "sup":
            torch.save(
                {
                    "epoch_sup": self.epoch_sup_current + 1,
                    "model_state_dict_sup": self.model.state_dict()
                    if "acs" not in self.config.model.lower()
                    else self.model.module.state_dict(),
                    "optimizer_state_dict_sup": self.optimizer_sup.state_dict(),
                    "scheduler_state_dict_sup": self.scheduler_sup.state_dict(),
                    "num_epoch_no_improvement_sup": self.num_epoch_no_improvement_sup,
                    "best_loss_sup": self.best_loss_sup,
                },
                os.path.join(self.config.model_path_save, "weights_sup{}.pt".format(suffix)),
            )
            print("Model Saved in {} \n".format(os.path.join(self.config.model_path_save, "weights_sup{}.pt".format(suffix))))

    def _loadparams(self, phase: str, **kwargs):
        """
        phase: "ss", "sup" or "both"
        """
        weight_dir = kwargs.get("dir", None)
        fresh_params = kwargs.get("fresh_params", False)

        # intialize params
        if phase == "ss" or phase == "both":

            if self.config.optimizer_ss.lower() == "sgd":
                self.optimizer_ss = torch.optim.SGD(
                    self.model.parameters(), self.config.lr_ss, momentum=0.9, weight_decay=0.0, nesterov=False
                )
            elif self.config.optimizer_ss.lower() == "adam":
                self.optimizer_ss = torch.optim.Adam(self.model.parameters(), self.config.lr_ss)

            if self.config.scheduler_ss.lower() == "reducelronplateau":
                self.scheduler_ss = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_ss, mode="min", factor=0.5, patience=self.config.patience_ss
                )
            elif self.config.scheduler_ss.lower() == "steplr":
                self.scheduler_ss = torch.optim.lr_scheduler.StepLR(self.optimizer_ss, step_size=int(self.config.patience_ss), gamma=0.5)

            elif self.config.scheduler_ss.lower() == "multisteplr":
                self.scheduler_ss = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer_ss, milestones=self.config.milestones, gamma=self.config.gamma
                )

        if phase == "sup" or phase == "both":

            if self.config.optimizer_sup.lower() == "sgd":
                self.optimizer_sup = torch.optim.SGD(
                    self.model.parameters(), self.config.lr_sup, momentum=0.9, weight_decay=0.0, nesterov=False
                )
            elif self.config.optimizer_sup.lower() == "adam":
                if hasattr(self.config, "eps_sup") and self.config.eps_sup is not None:
                    self.optimizer_sup = torch.optim.Adam(
                        self.model.parameters(),
                        self.config.lr_sup,
                        betas=(self.config.beta1_sup, self.config.beta2_sup),
                        eps=self.config.eps_sup,
                    )
                else:
                    self.optimizer_sup = torch.optim.Adam(
                        self.model.parameters(),
                        self.config.lr_sup,
                    )
            else:
                raise NotImplementedError

            if self.config.scheduler_sup.lower() == "reducelronplateau":
                self.scheduler_sup = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_sup, mode="min", factor=0.5, patience=self.config.patience_sup
                )

            elif self.config.scheduler_sup.lower() == "steplr":
                self.scheduler_sup = torch.optim.lr_scheduler.StepLR(self.optimizer_sup, step_size=int(self.config.patience_sup), gamma=0.5)

            elif self.config.scheduler_sup.lower() == "multisteplr":
                self.scheduler_sup = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer_sup, milestones=self.config.milestones, gamma=self.config.gamma
                )
            else:
                raise NotImplementedError

        if fresh_params:
            print("LOADING FRESH PARAMS")
            checkpoint = {}

        elif weight_dir:
            print("LOADING PARAMS FROM {} FOR PHASE {}".format(weight_dir, phase))
            checkpoint = torch.load(weight_dir, map_location=self.device)

            if phase == "ss" or phase == "both":
                self.optimizer_ss.load_state_dict(checkpoint["optimizer_state_dict_ss"])
                # RUN TIME ERROR FIX ON RESUME: https://github.com/jwyang/faster-rcnn.pytorch/issues/222
                # https://github.com/pytorch/pytorch/issues/2830
                #    for state in self.optimizer_ss.state.values():
                #        for k, v in state.items():
                #            if isinstance(v, torch.Tensor):
                #                state[k] = v.cuda()

                self.scheduler_ss.load_state_dict(checkpoint["scheduler_state_dict_ss"])

            if phase == "sup" or phase == "both":
                self.optimizer_sup.load_state_dict(checkpoint["optimizer_state_dict_sup"])
                # for state in self.optimizer_sup.state.values():
                #    for k, v in state.items():
                #        if isinstance(v, torch.Tensor):
                #            state[k] = v.cuda()
                self.scheduler_sup.load_state_dict(checkpoint["scheduler_state_dict_sup"])

        if phase == "ss" or phase == "both":

            self.num_epoch_no_improvement_ss = checkpoint.get("num_epoch_no_improvement_ss", 0)
            self.stats.training_losses_ss = checkpoint.get("training_losses_ss", [])
            self.stats.validation_losses_ss = checkpoint.get("validation_losses_ss", [])
            self.epoch_ss_check = checkpoint.get("epoch_ss", 0)
            self.best_loss_ss = checkpoint.get("best_loss_ss", 10000000000)

        if phase == "sup" or phase == "both":

            self.num_epoch_no_improvement_sup = checkpoint.get("num_epoch_no_improvement_sup", 0)
            self.stats.training_losses_sup = checkpoint.get("training_losses_sup", [])
            self.stats.validation_losses_sup = checkpoint.get("validation_losses_sup", [])
            self.epoch_sup_check = checkpoint.get("epoch_sup", 0)
            self.best_loss_sup = checkpoint.get("best_loss_sup", 10000000000)

    def get_stats(self):
        self.stats.get_statistics()

    def _get_dir_with_more_advanced_epoch(self, dir_a, dir_b, phase):

        check_a = torch.load(dir_a, map_location=self.device)
        check_b = torch.load(dir_b, map_location=self.device)
        if phase == "ss":
            epoch_a = check_a["epoch_ss"]
            epoch_b = check_b["epoch_ss"]
        if phase == "sup":
            epoch_a = check_a["epoch_sup"]
            epoch_b = check_b["epoch_sup"]

        return dir_a if epoch_a > epoch_b else dir_b

    @staticmethod
    def _convert_state_dict_between_archs(state_dict):
        # because of module arrangemnt made on ACS for easy return splits
        state_dict_copy = deepcopy(state_dict)
        for key in state_dict.keys():
            if "down" in key:
                tmp = key.split(".")
                tmp_keep = []
                tmp_keep.insert(0, tmp[0])
                tmp_keep.insert(1, tmp[-1])  # [down1, weight/bias/num_batches ...]
                tmp_middle = ".".join(tmp[1:-1])
                if tmp_middle == "maxpool_conv.1.double_conv.0":
                    tmp_middle = "conv1"
                elif tmp_middle == "maxpool_conv.1.double_conv.1":
                    tmp_middle = "bn1"
                elif tmp_middle == "maxpool_conv.1.double_conv.3":
                    tmp_middle = "conv2"
                elif tmp_middle == "maxpool_conv.1.double_conv.4":
                    tmp_middle = "bn2"
                else:
                    raise ValueError

                tmp_keep.insert(1, tmp_middle)
                new_key = ".".join(tmp_keep)
                print("REPLACING {} by {} in model state dict".format(key, new_key))
                state_dict_copy[new_key] = state_dict_copy[key]
                del state_dict_copy[key]
        return state_dict_copy


if __name__ == "__main__":
    pass
    # from ACSConv.acsconv.converters import ACSConverter

    # model = UNet(n_channels=1, n_classes=1, bilinear=True, apply_sigmoid_to_output=True)
    # a = model.state_dict()
    # a = []
    # for child_name, child in model.named_children():
    #    print(child)
    #    a.append(child_name)
    # model_acs = ACSConverter(model)
    # b = model_acs.state_dict()
    # for key, value in a.items():
    #    if key not in b:
    #        print("KEY DISCREPANCY")
    #    else:
    #        try:
    #            assert a[key] == b[key]
    #        except RuntimeError:
    #            assert torch.all(a[key].eq(b[key]))
    # torch.save(
    #    {"model_state_dict_unet_2d": model.state_dict(),}, "unet_2d.pt",
    # )
    # torch.save(
    #    {"model_state_dict_unet_acs": model_acs.state_dict(),}, "unet_acs.pt",
    # )
    # loaded_unet_2d = torch.load("unet_2d.pt", map_location="cuda")
    # loaded_unet_acs = torch.load("unet_acs.pt", map_location="cuda")
    # print(loaded_unet_2d["model_state_dict_unet_2d"].keys() == loaded_unet_acs["model_state_dict_unet_acs"].keys())
    # print(loaded_unet_2d["model_state_dict_unet_2d"].keys())
    # print(loaded_unet_acs["model_state_dict_unet_acs"].keys())

    # print(a == b)
    # b = []
    # print("/////////////////////////")
    # for child_name, child in model.named_children():
    #    print(child)
    #    b.append(child_name)
    #  print(a == b)
    # config = models_genesis_config()
    # dataset = Dataset(config.data_dir, train_val_test=(0.8, 0.2, 0)) # train_val_test is non relevant as will ve overwritten after
    # dataset.x_train_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.train_fold]
    # dataset.x_val_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.valid_fold]
    # dataset.x_test_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.test_fold] #Dont know in what sense they use this for
    # trainer_mg_replication = Trainer(config, dataset)
    # trainer_mg_replication.train_from_scratch_model_model_genesis_exact_replication()
    # trainer_mg_replication.add_hparams_to_writer()
    # trainer_mg_replication.get_stats()
