from finetune_config import FineTuneConfig
from config import models_genesis_config
from dataset import Dataset
from finetune import Trainer
from utils import make_dir


def resume_replication_of_results_pretrain(run_nr:int):
    
    config = models_genesis_config()
    config.override_dirs(run_nr)
    config.resume_ss = True
    config.scheduler_ss = "ReduceLROnPlateau"
    config.display()
    
    x_train_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.train_fold]
    x_val_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.valid_fold]
    x_test_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.test_fold] #Dont know in what sense they use this for
    files = [x_train_filenames, x_val_filenames, x_test_filenames]
    dataset = Dataset(config.data_dir, train_val_test=(0.8, 0.2, 0), file_names=files) # train_val_test is non relevant as is overwritten by files

    trainer_mg_replication = Trainer(config, dataset)
    trainer_mg_replication.load_model(from_latest_checkpoint=True) #still requires override dirs to find the specific checkpoint to resume from
    trainer_mg_replication.finetune_self_supervised()
    trainer_mg_replication.add_hparams_to_writer()
    trainer_mg_replication.get_stats()
    
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, dest="run", type=int)
    args = parser.parse_args()
    print("RESUMING RUN {}".format(args.run))
    resume_replication_of_results_pretrain(args.run)