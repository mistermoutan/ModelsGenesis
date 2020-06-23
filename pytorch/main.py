from finetune_config import FineTuneConfig
from config import models_genesis_config
from dataset import Dataset
from finetune import Trainer

#from memory_profiler import profile

# script to run experiments

# TODO BUILD CLI:
#       Their Replication:
#           - Replicate model genesis
#           - USe their pretrained weights to test on 5 datasets they provide
#       Me:
#           - Use their framework w/ different dataset (for scratch training) 
#           - Use the pretrrained models obtained to test on different datasets
#           - Move into modality , CT and MRI
#           

def replication_of_results_pretrain():

    #CUBE LOGIC
    # result replication, try to get the pretrained model
    config = models_genesis_config()
    dataset = Dataset(config.data_dir, train_val_test=(0.8, 0.2, 0)) # train_val_test is non relevant as will ve overwritten after
    dataset.x_train_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.train_fold]
    dataset.x_val_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.valid_fold]
    dataset.x_test_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.test_fold] #Dont know in what sense they use this for
    trainer_mg_replication = Trainer(config, dataset)
    trainer_mg_replication.train_from_scratch_model_genesis_exact_replication()
    trainer_mg_replication.add_hparams_to_writer()
    trainer_mg_replication.get_stats()  
    


if __name__ == "__main__":
    
    print("STARTING REPLICATION OF RESULTS EXPERIMENT")
    replication_of_results_pretrain()
    print("FINISHED REPLICATION OF RESULTS EXPERIMENT")

#

