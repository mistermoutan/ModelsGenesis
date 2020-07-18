import time
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os
from utils import make_dir
from collections import OrderedDict
import json
from finetune_config import FineTuneConfig
from config import models_genesis_config
from dataset import Dataset

from datasets_pytorch import DatasetsPytorch

"""
Class to benchmark performance of methods
HOW TO USE:
1. Create instance of class once training is initiated, see the constructor for the arguments you need to pass
2. During training append the values to the relevant lists of the constructor or find proper ways of completing them. 
3. At the end of training call the get_statistics method (stat.get_statistics()) and BOOM done just like it was magic
"""


class Statistics:
    def __init__(self, config: FineTuneConfig or models_genesis_config, dataset: Dataset or DatasetsPytorch):
        """
        save_directory: directory where you want to save the stats
        """

        self.config = config
        self.dataset = dataset
        self.multiple_datasets = True if isinstance(self.dataset, list) else False

        self.start_time = time.time()
        self.save_directory = config.stats_dir
        make_dir(self.save_directory)

        # if self.config.self_supervised:
        self.training_losses_ss = []
        self.validation_losses_ss = []
        self.avg_training_loss_per_epoch_ss = []
        self.avg_validation_loss_per_epoch_ss = []
        self.iterations_ss = []
        self.stopped_early_ss = False
        self.comment_ss = None

        # self.threshold = config.threshold
        self.training_losses_sup = []
        self.validation_losses_sup = []
        self.avg_training_loss_per_epoch_sup = []
        self.avg_validation_loss_per_epoch_sup = []
        self.iterations_sup = []
        self.stopped_early_sup = False
        self.comment_sup = None

    def get_statistics(self):

        self.end_time = time.time()
        self.build_stats_dictionary()
        # self.save_graphs()
        self.save_log_book()
        self.save_arrays()

    def build_stats_dictionary(self):
        """ builds dictionary of statistics (self.stats), saves it as a pickle"""

        self.stats = OrderedDict()
        self.stats["date"] = time.strftime("%c")
        self.stats["training_time"] = str(datetime.timedelta(seconds=self.end_time - self.start_time))
        self.stats["task"] = self.config.task

        self.stats["dataset_dir"] = self.config.data_dir  # will give us names of datasets used
        if not self.multiple_datasets:
            self.stats["tr_val_ts_split"] = self.dataset.tr_val_ts_split
        else:
            self.stats["tr_val_ts_split"] = self.dataset[0].tr_val_ts_split

        # self.stats["training_files"] = self.dataset.x_train_filenames
        # self.stats["validation_files"] = self.dataset.x_val_filenames
        # self.stats["test_files"] = self.dataset.x_test_filenames

        self.stats["self_supervised_used"] = True if self.config.self_supervised else False

        if self.config.self_supervised:
            self.stats["epochs_ss"] = self.config.nb_epoch_ss
            self.stats["batch_size_ss"] = self.config.batch_size_ss
            self.stats["initial_lr_ss"] = self.config.lr_ss
            self.stats["loss_function_ss"] = self.config.loss_function_ss
            self.stats["optimizer_ss"] = self.config.optimizer_ss
            self.stats["scheduler_ss"] = self.config.scheduler_ss
            self.stats["stopped_early_ss"] = self.stopped_early_ss
            self.stats["iterations_ss"] = list(set(self.iterations_ss))
            self.stats["comment_ss"] = self.comment_ss

        self.stats["epochs_sup"] = self.config.nb_epoch_sup
        self.stats["batch_size_sup"] = self.config.batch_size_sup
        self.stats["initial_lr_sup"] = self.config.lr_sup
        self.stats["loss_function_sup"] = self.config.loss_function_sup
        self.stats["optimizer_sup"] = self.config.optimizer_sup
        self.stats["scheduler_sup"] = self.config.scheduler_sup
        self.stats["stopped_early_sup"] = self.stopped_early_sup
        self.stats["iterations_sup"] = list(set(self.iterations_sup))
        self.stats["comment_sup"] = self.comment_sup

        self.write_pickle(self.stats, self.save_directory, "stats.pickle")

    def save_arrays(self):
        """ saves arrays in dict, may be relevant for future comparisons"""

        dict_of_arrays = OrderedDict()
        # dict_of_arrays["training_files"] = self.dataset.x_train_filenames
        # dict_of_arrays["validation_files"] = self.dataset.x_val_filenames
        # ict_of_arrays["test_files"] = self.dataset.x_test_filenames

        if self.config.self_supervised:
            dict_of_arrays["avg_training_loss_per_epoch_ss"] = self.avg_training_loss_per_epoch_ss
            dict_of_arrays["avg_validation_loss_per_epoch_ss"] = self.avg_validation_loss_per_epoch_ss

        dict_of_arrays["avg_training_loss_per_epoch_sup"] = self.avg_training_loss_per_epoch_sup
        dict_of_arrays["avg_validation_loss_per_epoch_sup"] = self.avg_validation_loss_per_epoch_sup

        self.write_pickle(dict_of_arrays, self.save_directory, "arrays.pickle")

    def save_graphs(self):

        if self.config.self_supervised and isinstance(self.config.nb_epoch_ss, int):
            # plot avg training loss per epoch
            x_axis = np.arange(self.config.nb_epoch_ss)
            figure = plt.figure()
            plt.plot(x_axis, self.avg_training_loss_per_epoch_ss)
            plt.title("Avg Training Loss per epoch SS")
            figure.savefig(self.save_directory + "avg_training_loss_per_epoch_ss.png")

            # plot avg validation loss per epoch
            x_axis = np.arange(self.config.nb_epoch_ss)
            figure = plt.figure()
            plt.plot(x_axis, self.avg_validation_loss_per_epoch_ss)
            plt.title("Avg Validation Loss per epoch ")
            figure.savefig(self.save_directory + "avg_validation_loss_per_epoch_ss.png")

        if isinstance(self.config.nb_epoch_sup, int):
            # plot avg training loss per epoch
            x_axis = np.arange(self.config.nb_epoch_sup)
            figure = plt.figure()
            plt.plot(x_axis, self.avg_training_loss_per_epoch_sup)
            plt.title("Avg Training Loss per epoch SUP")
            figure.savefig(self.save_directory + "avg_training_loss_per_epoch_sup.png")

            # plot avg validation loss per epoch
            x_axis = np.arange(self.config.nb_epoch_sup)
            figure = plt.figure()
            plt.plot(x_axis, self.avg_validation_loss_per_epoch_sup)
            plt.title("Avg Validation Loss per epoch")
            figure.savefig(self.save_directory + "avg_validation_loss_per_epoch_sup.png")

    def save_log_book(self, save_json=True):

        if save_json:
            with open(self.save_directory + "log.json", "w") as logbook:
                json.dump(self.stats, logbook)
        else:
            with open(self.save_directory + "log.txt", "w") as logbook:
                for key, value in self.stats.items():
                    logbook.write("{0} : {1}  \n ".format(key, value))

    @staticmethod
    def write_pickle(f, path, fname):
        with open(path + fname, "wb") as handle:
            pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)
