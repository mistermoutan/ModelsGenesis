import time
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os

"""
Class to benchmark performance of methods
HOW TO USE:
1. Create instance of class once training is initiated, see the constructor for the arguments you need to pass
2. During training append the values to the relevant lists of the constructor or find proper ways of completing them. 
3. At the end of training call the get_statistics method (stat.get_statistics()) and BOOM done just like it was magic
"""

class Statistics:

    def __init__(self,epochs: int, batch_size:int,  save_directory:str = "stats/"):
        """
        epochs: number of epochs
        save_directory: directory where you want to save the stats
        """
        
        self.start_time = time.time()
        self.save_directory = save_directory
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.training_losses = []
        self.validation_losses = []
        self.avg_training_loss_per_epoch = [] 
        self.avg_validation_loss_per_epoch = []
        
        self.iterations = []
        self.stopped_early = False
        self.comment = None
        self.tr_val_ts_split = None
        self.task = None
        self.initial_lr = None

    def get_statistics(self):

        self.end_time = time.time() #for tracking training time
        self.build_stats_dictionary()
        self.save_graphs()
        self.save_log_book()
        self.save_arrays()

    def build_stats_dictionary(self):
        """ builds dictionary of statistics (self.stats), saves it as a pickle"""
            
        self.stats = {}
        self.stats["date"] = time.strftime("%c")
        self.stats["batch_size"] = self.batch_size
        self.stats["training_time"] = str(datetime.timedelta(seconds=self.end_time - self.start_time))
        self.stats["stopped_early"] = self.stopped_early
        self.stats["iterations"] = set(self.iterations)
        if self.comment:
            self.stats["description"] = self.comment
        self.stats["tr_val_ts_split"] = self.tr_val_ts_split
        self.stats["task"] = self.task
        self.stats["initial_lr"] = self.initial_lr
                
        self.write_pickle(self.stats,self.save_directory,"stats.pickle")

    def save_arrays(self):
        """ saves arrays in dict, may be relevant for future comparisons"""

        dict_of_arrays = {}
        dict_of_arrays["avg_training_loss_per_epoch"] = self.avg_training_loss_per_epoch
        dict_of_arrays["avg_validation_loss_per_epoch"] = self.avg_validation_loss_per_epoch
        self.write_pickle(dict_of_arrays,self.save_directory,"arrays.pickle")

    def save_graphs (self):

        #plot avg training loss per epoch
        x_axis = np.arange(self.epochs)
        figure = plt.figure()
        plt.plot(x_axis,self.avg_training_loss_per_epoch)
        plt.title("Avg Training Loss per epoch ")
        figure.savefig(self.save_directory + "avg_training_loss_per_epoch.png")

        #plot avg validation loss per epoch
        x_axis = np.arange(self.epochs)
        figure = plt.figure()
        plt.plot(x_axis,self.avg_validation_loss_per_epoch)
        plt.title("Avg Validation Loss per epoch ")
        figure.savefig(self.save_directory + "avg_validation_loss_per_epoch.png")

    def save_log_book (self):
        with open(self.save_directory + "log.txt",'w') as logbook:
            for key,value in self.stats.items():
                logbook.write("{0} : {1}  \n " .format(key,value))
    
    @staticmethod
    def write_pickle (f, path, fname):
        with open(path + fname, 'wb') as handle:
            pickle.dump(f, handle, protocol = pickle.HIGHEST_PROTOCOL)