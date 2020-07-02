from finetune_config import FineTuneConfig
from config import models_genesis_config
from dataset import Dataset
from finetune import Trainer
from utils import make_dir

# script to run experiments

# TODO BUILD CLI:
#       Their Replication:
#           - Replicate model genesis
#           - USe their pretrained weights to test on 5 datasets they provide
#       Me:
#           - Use their framework w/ different dataset (for scratch training) 
#           - Use the pretrrained models obtained to test on different datasets
#           - Move into modality , CT and MRI


def replication_of_results_pretrain():

    config = models_genesis_config()
    config.display()
    x_train_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.train_fold]
    x_val_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.valid_fold]
    x_test_filenames = ["bat_32_s_64x64x32_" + str(i) + ".npy" for i in config.test_fold] #Dont know in what sense they use this for
    files = [x_train_filenames, x_val_filenames, x_test_filenames]
    dataset = Dataset(config.data_dir, train_val_test=(0.8, 0.2, 0), file_names=files) # train_val_test is non relevant as is overwritten by files

    trainer_mg_replication = Trainer(config, dataset)
    trainer_mg_replication.load_model(from_scratch=True)
    trainer_mg_replication.finetune_self_supervised()
    trainer_mg_replication.add_hparams_to_writer()
    trainer_mg_replication.get_stats()

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
    parser.add_argument("--command", required=True, dest="command", type=str)
    parser.add_argument("--run", required=False, dest="run", default=None, type=int)
    args = parser.parse_args()
    if args.command == "replicate_model_genesis_pretrain":
        print("STARTING REPLICATION OF RESULTS EXPERIMENT")
        replication_of_results_pretrain()
    elif args.command == "resume_model_genesis_pretrain":
        assert args.run is not None, "You have to specify which --run to resume (int)"
        print("RESUMING REPLICATION OF RESULTS EXPERIMENT")
        print("RESUMING RUN {}".format(args.run))
        resume_replication_of_results_pretrain(args.run)