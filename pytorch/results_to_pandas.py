import os
import pandas as pd

class ResultCollector:

    def __init__(self):
        pass

    def _get_task_names_and_corresponding_metric_files(self):

        res = [] # [ (task_dir, file) ...]
        for root, dirs, files in os.walk("test_results"):
            if not files:
                continue
            else:
                root_split = root.split("/")
                task_dir = "/".join(i for i in root_split[1:])
                if task_dir == "":
                    print("SHOULD BE PRINTING THE PROVIDED WEIGHTS")
                    print("\n", "ROOT", root, "\n", "DIRS ", dirs, "\n", "FILES", files, "\n")
                    continue
                res.append((task_dir, files)
        return res
    
    def _build_dataframe(self):

        task_names_and_files = self._get_task_names_and_corresponding_metric_files()
        
        task_names = []
        target_datasets = [] 

        full_cubes_dice_training = []
        full_cubes_dice_testing = []

        full_cubes_jaccard_training = []
        full_cubes_jaccard_testing = []

        mini_cubes_dice_training = []
        mini_cubes_dice_testing = []

        mini_cubes_jaccard_training = []
        mini_cubes_jaccard_testing = []

        for idx, (task_dir, _file) in enumerate(task_names_and_files):
            with open(os.path.join("test_results", task_dir, _file), "r") as f:
                results_dict = json.load(f) # 

            for dataset_name, mini_and_full_cubes_dict in results_dict.items():
                
                task_names.append(task_dir)
                target_datasets.append(dataset_name)

                full_cubes_dice_training.append(mini_and_full_cubes_dict["full_cubes"]["dice_train"])
                full_cubes_dice_testing.append(mini_and_full_cubes_dict["full_cubes"]["dice_test"])
                full_cubes_jaccard_training.append(mini_and_full_cubes_dict["full_cubes"]["jaccard_train"]
                full_cubes_jaccard_testing.append(mini_and_full_cubes_dict["full_cubes"]["jaccard_test"]

                mini_cubes_dice_training.append(mini_and_full_cubes_dict["mini_cubes"]["dice_train"]
                mini_cubes_dice_testing.append(mini_and_full_cubes_dict["mini_cubes"]["dice_test"]
                mini_cubes_jaccard_training.append(mini_and_full_cubes_dict["mini_cubes"]["jaccard_train"]
                mini_cubes_jaccard_testing.append(mini_and_full_cubes_dict["mini_cubes"]["jaccard_test"]

        df = pd.DataFrame({
            "task_name": task_names,
            "target_dataset": target_datasets,
            "full_cubes_dice_training": full_cubes_dice_training, 
            "full_cubes_jaccard_training": full_cubes_jaccard_training,
            "full_cubes_dice_testing": full_cubes_dice_testing,
            "full_cubes_jaccard_testing": full_cubes_jaccard_testing,
            "mini_cubes_dice_training": mini_cubes_dice_training,
            "mini_cubes_jaccard_training": mini_cubes_jaccard_training,
            "mini_cubes_dice_testing": mini_cubes_dice_testing,
            "mini_cubes_jaccard_testing": mini_cubes_jaccard_testing
            })