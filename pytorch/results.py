import os
import pandas as pd
import json


class ResultHanlder:
    def __init__(self):
        pass

    def get_df(self):
        self._build_dataframe()
        self._add_experimet_id_and_task_name()
        self._add_experiment_details()
        return self.df

    def _get_task_names_and_corresponding_metric_files(self):

        res = []  # [ (task_dir, file) ...]
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
                assert isinstance(files, list)
                res.append((task_dir, files[0]))
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
            # print(task_dir)
            # print(_file)
            with open(os.path.join("test_results", task_dir, _file), "r") as f:
                results_dict = json.load(f)  #

            for dataset_name, mini_and_full_cubes_dict in results_dict.items():

                task_names.append(task_dir)
                target_datasets.append(dataset_name)

                try:
                    full_cubes_dice_training.append(mini_and_full_cubes_dict["full_cubes"]["dice_train"])
                    full_cubes_dice_testing.append(mini_and_full_cubes_dict["full_cubes"]["dice_test"])
                    full_cubes_jaccard_training.append(mini_and_full_cubes_dict["full_cubes"]["jaccard_train"])
                    full_cubes_jaccard_testing.append(mini_and_full_cubes_dict["full_cubes"]["jaccard_test"])
                except KeyError:
                    # not all full_cubes were tested
                    full_cubes_dice_training.append("nAn")
                    full_cubes_dice_testing.append("nAn")
                    full_cubes_jaccard_training.append("nAn")
                    full_cubes_jaccard_testing.append("nAn")

                mini_cubes_dice_training.append(mini_and_full_cubes_dict["mini_cubes"]["dice_train"])
                mini_cubes_dice_testing.append(mini_and_full_cubes_dict["mini_cubes"]["dice_test"])
                mini_cubes_jaccard_training.append(mini_and_full_cubes_dict["mini_cubes"]["jaccard_train"])
                mini_cubes_jaccard_testing.append(mini_and_full_cubes_dict["mini_cubes"]["jaccard_test"])

        self.df = pd.DataFrame(
            {
                "task_name_with_run": task_names,
                "target_dataset": target_datasets,
                "full_cubes_dice_training": full_cubes_dice_training,
                "full_cubes_jaccard_training": full_cubes_jaccard_training,
                "full_cubes_dice_testing": full_cubes_dice_testing,
                "full_cubes_jaccard_testing": full_cubes_jaccard_testing,
                "mini_cubes_dice_training": mini_cubes_dice_training,
                "mini_cubes_jaccard_training": mini_cubes_jaccard_training,
                "mini_cubes_dice_testing": mini_cubes_dice_testing,
                "mini_cubes_jaccard_testing": mini_cubes_jaccard_testing,
            }
        )

    def _add_experimet_id_and_task_name(self):

        exp_map = dict()
        exp_num = 0
        exp_num_list = []
        task_name_list = []
        run_number_list = []
        for idx, row_series in self.df.iterrows():
            task_name_with_run = row_series.task_name_with_run
            task_name = "/".join(i for i in task_name_with_run.split("/")[:-1])
            run_number = task_name_with_run.split("/")[-1]
            run_number_list.append(run_number)
            task_name_list.append(task_name)
            if task_name in exp_map:
                exp_num_list.append(exp_map[task_name])
            else:
                exp_map[task_name] = exp_num
                exp_num_list.append(exp_num)
                exp_num += 1

        self.df["experiment_id"] = exp_num_list
        self.df["task_name"] = task_name_list
        self.df["run_number"] = run_number_list

    def _add_experiment_details(self):

        # experiments_category = []
        experiment_descriptions = []  # -> change to category
        modalities_list = []
        uses_genesis_transforms = []
        uses_acs_paper_transforms = []
        from_scratch = []
        pretrained_on_ct = []
        pretrained_on_mri = []
        cross_modality = []
        pretrained_weights = []
        model = []
        pretrain_ss_on_same_dataset_before_supervised = []

        for idx, row_series in self.df.iterrows():

            task_name = row_series.task_name

            if "FROM_PROVIDED_WEIGHTS_SS_AND_SUP_lidc_VNET_MG/with_self_supervised" == task_name:

                # experiments_category.append("model_genesis_paper_validation")
                experiment_descriptions.append("model_genesis_paper_validation_on_lidc_ss_sup")
                modalities_list.append("ct")
                uses_genesis_transforms.append(True)
                uses_acs_paper_transforms.append(False)
                from_scratch.append(True)  # ?
                pretrained_on_ct.append(True)
                pretrained_on_mri.append(False)
                cross_modality.append("ct_to_ct")
                pretrained_weights.append("model_genesis")  # so now from_scratch = True?
                model.append("VNET_MG")
                pretrain_ss_on_same_dataset_before_supervised.append(True)

            elif "FROM_PROVIDED_WEIGHTS_SUP_ONLY_lidc_VNET_MG/only_supervised" == task_name:

                experiment_descriptions.append("model_genesis_paper_validation_on_lidc_sup")
                modalities_list.append("ct")
                uses_genesis_transforms.append(True)
                uses_acs_paper_transforms.append(False)
                from_scratch.append(True)
                pretrained_on_ct.append(True)
                pretrained_on_mri.append(False)
                cross_modality.append("ct_to_ct")
                pretrained_weights.append("model_genesis")
                model.append("VNET_MG")
                pretrain_ss_on_same_dataset_before_supervised.append(False)  # model genesis weights were pretrained on luna16

            elif task_name in (
                "FROM_PROVIDED_WEIGHTS_task01_sup_VNET_MG/only_supervised",
                "FROM_PROVIDED_WEIGHTS_task04_sup_VNET_MG/only_supervised",
            ):

                experiment_descriptions.append("model_genesis_weights_generalization_to_other_datasets")
                modalities_list.append("mri")
                uses_genesis_transforms.append(True)
                uses_acs_paper_transforms.append(False)
                from_scratch.append(True)
                pretrained_on_ct.append(True)
                pretrained_on_mri.append(False)
                cross_modality.append("ct_to_mri")
                pretrained_weights.append("model_genesis")
                model.append("VNET_MG")
                pretrain_ss_on_same_dataset_before_supervised.append(False)

            elif "FROM_SCRATCH_cellari_heart_sup_10_192" in task_name:
                experiment_descriptions.append("unet_2D_vs_3D_vs_ACS_on_cellari_dataset")
                modalities_list.append("mri")
                uses_genesis_transforms.append(False)
                uses_acs_paper_transforms.append(False)
                from_scratch.append(True)
                pretrained_on_ct.append(False)
                pretrained_on_mri.append(False)
                cross_modality.append(False)
                pretrained_weights.append(False)
                pretrain_ss_on_same_dataset_before_supervised.append(False)

                if "unet_acs" in task_name.lower():
                    model.append("UNET_ACS")
                elif "unet_2d" in task_name.lower():
                    model.append("UNET_2D")
                elif "unet_3d" in task_name.lower():
                    model.append("UNET_3D")

            elif "FROM_SCRATCH_cellari_heart_sup" in task_name and "FROM_SCRATCH_cellari_heart_sup_10_192" not in task_name:
                experiment_descriptions.append("unet_2D_vs_3D_vs_ACS_on_cellari_dataset_small_cubes")
                modalities_list.append("mri")
                uses_genesis_transforms.append(False)
                uses_acs_paper_transforms.append(False)
                from_scratch.append(True)
                pretrained_on_ct.append(False)
                pretrained_on_mri.append(False)
                cross_modality.append(False)
                pretrained_weights.append(False)
                pretrain_ss_on_same_dataset_before_supervised.append(False)

                if "unet_acs" in task_name.lower():
                    model.append("UNET_ACS")
                elif "unet_2d" in task_name.lower():
                    model.append("UNET_2D")
                elif "unet_3d" in task_name.lower():
                    model.append("UNET_3D")

            elif "FROM_SCRATCH_lidc_VNET_MG/only_supervised" == task_name:  # only 2 runs?

                experiment_descriptions.append("from_scratch_on_lidc_to_see_if_thre_is_advantage_in_pretraining")
                modalities_list.append("ct")
                uses_genesis_transforms.append(False)
                uses_acs_paper_transforms.append(False)
                from_scratch.append(True)
                pretrained_on_ct.append(False)
                pretrained_on_mri.append(False)
                cross_modality.append(False)
                pretrained_weights.append(False)
                model.append("VNET_MG")
                pretrain_ss_on_same_dataset_before_supervised.append(False)

            elif "FROM_SCRATCH_task04_sup_VNET_MG/only_supervised" == task_name:
                experiment_descriptions.append("from_scratch_on_task04_to_see_if_there_is_advantage_in_pretraining")
                modalities_list.append("mri")
                uses_genesis_transforms.append(False)
                uses_acs_paper_transforms.append(False)
                from_scratch.append(True)
                pretrained_on_ct.append(False)
                pretrained_on_mri.append(False)
                cross_modality.append(False)
                pretrained_weights.append(False)
                model.append("VNET_MG")
                pretrain_ss_on_same_dataset_before_supervised.append(False)

            elif task_name in (
                "FROM_pretrained_weights/GENESIS_REPLICATION_PRETRAIN_MODEL/run_6__lidc_VNET_MG/only_supervised",
                "FROM_pretrained_weights/GENESIS_REPLICATION_PRETRAIN_MODEL/run_5__lidc_VNET_MG/only_supervised",
                "FROM_pretrained_weights/GENESIS_REPLICATION_PRETRAIN_MODEL/run_7__lidc_VNET_MG/only_supervised",
            ):  # only 2 runs on run 5? possibly because run5 was a fuck up
                experiment_descriptions.append("model_genesis_validation_replicate_given_weights")
                modalities_list.append("ct")
                uses_genesis_transforms.append(True)
                uses_acs_paper_transforms.append(False)
                from_scratch.append(False)
                pretrained_on_ct.append(True)
                pretrained_on_mri.append(False)
                cross_modality.append("ct_to_ct")
                pretrained_weights.append(False)
                model.append("VNET_MG")
                pretrain_ss_on_same_dataset_before_supervised.append(False)

            elif task_name in (
                "FROM_pretrained_weights/GENESIS_REPLICATION_PRETRAIN_MODEL/run_5__lidc_VNET_MG/with_self_supervised",
                "FROM_pretrained_weights/GENESIS_REPLICATION_PRETRAIN_MODEL/run_6__lidc_VNET_MG/with_self_supervised",
                "FROM_pretrained_weights/GENESIS_REPLICATION_PRETRAIN_MODEL/run_7__lidc_VNET_MG/with_self_supervised",
            ):
                experiment_descriptions.append("model_genesis_validation_replicate_given_weights")
                modalities_list.append("ct")
                uses_genesis_transforms.append(True)
                uses_acs_paper_transforms.append(False)
                from_scratch.append(False)
                pretrained_on_ct.append(True)
                pretrained_on_mri.append(False)
                cross_modality.append("ct_to_ct")
                pretrained_weights.append(False)
                model.append("VNET_MG")
                pretrain_ss_on_same_dataset_before_supervised.append(True)

            elif task_name in (
                "FROM_pretrained_weights/GENESIS_REPLICATION_PRETRAIN_MODEL/run_6__task01_sup_VNET_MG/only_supervised",
                "FROM_pretrained_weights/GENESIS_REPLICATION_PRETRAIN_MODEL/run_7__task01_sup_VNET_MG/only_supervised",
            ):

                experiment_descriptions.append("model_genesis_validation_replicate_given_weights")
                modalities_list.append("mri")
                uses_genesis_transforms.append(True)
                uses_acs_paper_transforms.append(False)
                from_scratch.append(False)
                pretrained_on_ct.append(True)
                pretrained_on_mri.append(False)
                cross_modality.append("ct_to_mri")
                pretrained_weights.append(False)
                model.append("VNET_MG")
                pretrain_ss_on_same_dataset_before_supervised.append(False)

            elif task_name in (
                "FROM_pretrained_weights/PRETRAIN_MG_FRAMEWORK_task01_ss_VNET_MG/run_1__task01_sup_VNET_MG/only_supervised",
                "FROM_pretrained_weights/PRETRAIN_MG_FRAMEWORK_task01_ss_VNET_MG/run_2__task01_sup_VNET_MG/only_supervised",
                "FROM_pretrained_weights/PRETRAIN_MG_FRAMEWORK_task04_ss_VNET_MG/run_2__task04_sup_VNET_MG/only_supervised",
            ):

                experiment_descriptions.append("pretrain_with_mg_framework")
                modalities_list.append("mri")
                uses_genesis_transforms.append(True)
                uses_acs_paper_transforms.append(False)
                from_scratch.append(False)
                pretrained_on_ct.append(False)
                pretrained_on_mri.append(True)
                cross_modality.append("mri_to_mri_same_dataset")
                pretrained_weights.append(False)
                model.append("VNET_MG")
                pretrain_ss_on_same_dataset_before_supervised.append(
                    False
                )  # because this is for doing quick ss, not full lenght model genesis. #TODO CONFIRM
            elif task_name in (
                "FROM_pretrained_weights/PRETRAIN_MG_FRAMEWORK_task01_ss_VNET_MG/run_1__task04_sup_VNET_MG/only_supervised",
                "FROM_pretrained_weights/PRETRAIN_MG_FRAMEWORK_task04_ss_VNET_MG/run_2__task01_sup_VNET_MG/only_supervised",
            ):
                experiment_descriptions.append("pretrain_with_mg_framework")
                modalities_list.append("mri")
                uses_genesis_transforms.append(True)
                uses_acs_paper_transforms.append(False)
                from_scratch.append(False)
                pretrained_on_ct.append(False)
                pretrained_on_mri.append(True)
                cross_modality.append("mri_to_mri")
                pretrained_weights.append(False)
                model.append("VNET_MG")
                pretrain_ss_on_same_dataset_before_supervised.append(False)

            elif (
                task_name
                == "FROM_pretrained_weights/PRETRAIN_MG_FRAMEWORK_task04_ss_VNET_MG/run_2__task04_sup_VNET_MG/with_self_supervised"
            ):
                experiment_descriptions.append("pretrain_with_mg_framework")
                modalities_list.append("mri")
                uses_genesis_transforms.append(True)
                uses_acs_paper_transforms.append(False)
                from_scratch.append(False)
                pretrained_on_ct.append(False)
                pretrained_on_mri.append(True)
                cross_modality.append("mri_to_mri_same_dataset")
                pretrained_weights.append(False)
                model.append("VNET_MG")
                pretrain_ss_on_same_dataset_before_supervised.append(True)

            elif "FROM_pretrained_weights/PRETRAIN_MG_FRAMEWORK_task07_ss_task08_ss" in task_name:
                experiment_descriptions.append("pretrain_with_mg_framework")
                modalities_list.append("ct")
                uses_genesis_transforms.append(True)
                uses_acs_paper_transforms.append(False)
                from_scratch.append(False)
                pretrained_on_ct.append(True)
                pretrained_on_mri.append(False)
                cross_modality.append("2cts_to_1ct")
                pretrained_weights.append(False)
                model.append("VNET_MG")
                pretrain_ss_on_same_dataset_before_supervised.append(False)

            else:
                print(task_name)

        print(len(self.df))
        print(len(experiment_descriptions))
        self.df["experiments_description"] = experiment_descriptions
        self.df["target_modality"] = modalities_list
        self.df["uses_genesis_transforms"] = uses_genesis_transforms
        self.df["uses_acs_paper_transforms"] = uses_acs_paper_transforms
        self.df["from_scratch"] = from_scratch
        self.df["pretrained_on_ct"] = pretrained_on_ct
        self.df["pretrained_on_mri"] = pretrained_on_mri
        self.df["modality_to_modality"] = cross_modality
        self.df["official_pretrained_weights"] = pretrained_weights
        self.df["model"] = model
        self.df["performed_quick_ss_pre_sup"] = pretrain_ss_on_same_dataset_before_supervised

    def _get_condensed_df(self):
        pass


if __name__ == "__main__":

    pass