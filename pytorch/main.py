from finetune_config import FineTuneConfig
from config import models_genesis_config
from dataset import Dataset
from finetune import Trainer

# script to run experiments

if __name__ == "__main__":

    print("STARTING REPLICATION OF RESULTS EXPERIMENT")
    # result replication, try to get the pretrained model
    config = models_genesis_config()
    dataset = Dataset(config.data_dir, train_val_test=(0.8, 0.2, 0)) # train_val_test is non relevant as will ve overwritten after
    dataset.x_train_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.train_fold]
    dataset.x_val_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.valid_fold]
    dataset.x_test_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.test_fold] #Dont know in what sense they use this for
    trainer_mg_replication = Trainer(config, dataset)
    trainer_mg_replication.train_from_scratch_model_model_genesis_exact_replication()
    trainer_mg_replication.get_stats()
    print("FINISHED REPLICATION OF RESULTS EXPERIMENT")


#





