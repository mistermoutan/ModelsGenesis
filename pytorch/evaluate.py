from dice_loss import DiceLoss
import numpy as np
from sklearn.metrics import jaccard_score
from dice_loss import DiceLoss
from unet3d import UNet3D
import json
from copy import deepcopy
from collections import defaultdict
import os
import torch


from dataset import Dataset
from utils import *
from full_cube_segmentation import FullCubeSegmentator
from finetune import Trainer


class Tester:
    def __init__(self, config, dataset, test_all=True):

        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_results_dir = os.path.join("test_results/", self.config.task_dir)
        make_dir(self.test_results_dir)
        self.test_all = test_all

        self.trainer = Trainer(config=self.config, dataset=None)  # instanciating trainer to load and access model
        self.trainer.load_model(from_path=True, path=os.path.join(self.config.model_path_save, "weights_sup.pt"), phase="sup")
        self.model = self.trainer.model

        self.metric_dict = dict()
        self.metric_dict_unused = dict()

    def test_segmentation(self):

        if isinstance(self.dataset, list):
            for dataset in self.dataset:
                self._test_dataset(dataset)
                self._test_on_full_cubes(dataset)
        else:
            self._test_dataset(self.dataset)
            self._test_on_full_cubes(self.dataset)

        file_nr = 0
        while os.path.isfile(os.path.join(self.test_results_dir, "test_results{}.json".format(file_nr))):
            file_nr += 1

        # to check if testing remains consistent
        with open(os.path.join(self.test_results_dir, "test_results{}.json".format(file_nr)), "w") as f:
            json.dump(self.metric_dict, f)

        if self.test_all:
            # set of dataset data dirs
            remaining_datasets = get_unused_datasets(self.dataset)
            if len(remaining_datasets) > 0:
                for dataset in remaining_datasets:
                    # test on all other unseen datasets
                    full_test_ds = Dataset(data_dir=dataset, train_val_test=(0, 0, 1), file_names=None)
                    if not full_test_ds.has_target:
                        continue
                    self._test_dataset(full_test_ds, unused=True)

                with open(os.path.join(self.test_results_dir, "test_results_unused{}.json".format(file_nr)), "w") as f:
                    json.dump(self.metric_dict_unused, f)

    def _test_dataset(self, dataset, unused=False):
        # mini cubes settting
        dataset_dict = dict()

        # for model_file in self.model_weights_saved:
        #    dataset_dict.setdefault(model_file, {})
        #    self._load_model(model_file)
        #    if self.continue_to_testing is False:
        #        continue

        dataset_dict.setdefault("mini_cubes", {})
        jaccard = []
        dice = []

        # MAKE TRAINING VS TESTING differentiation on metrics
        # raise ValueError

        if dataset.x_test_filenames_original != []:
            previous_len = len(dataset.x_val_filenames_original)
            previous_val_filenames = deepcopy(dataset.x_val_filenames_original)
            dataset.x_val_filenames_original.extend(dataset.x_test_filenames_original)
            dataset.reset()
            assert len(dataset.x_val_filenames_original) == previous_len + len(dataset.x_test_filenames_original)
            assert previous_val_filenames != dataset.x_val_filenames_original

        with torch.no_grad():
            self.model.eval()
            while True:
                # if dataset.x_test_filenames:
                #    print("USINGdataset.x_test_filenames)

                # batch size > 1 plus random shuffle of indexes in Dataset results in no having the exact same
                # testins result each time as batches are flattened and act "as one"
                # so you would be giving the metrics different tensors to work with
                x, y = dataset.get_val(batch_size=1, return_tensor=True)
                if x is None:
                    break
                x, y = x.float().to(self.device), y.float().to(self.device)
                pred = self.model(x)
                x = self._make_pred_mask_from_pred(pred)

                dice.append(float(DiceLoss.dice_loss(x, y, return_loss=False)))

                x_flat = x[:, 0].contiguous().view(-1)
                y_flat = y[:, 0].contiguous().view(-1)
                x_flat = x_flat.cpu()
                y_flat = y_flat.cpu()
                jaccard.append(jaccard_score(y_flat, x_flat))
            dataset.reset()

        avg_jaccard = sum(jaccard) / len(jaccard)
        avg_dice = sum(dice) / len(dice)
        dataset_dict["mini_cubes"]["jaccard_test"] = avg_jaccard
        dataset_dict["mini_cubes"]["dice_test"] = avg_dice

        jaccard = []
        dice = []
        with torch.no_grad():
            self.model.eval()
            while True:
                x, y = dataset.get_train(batch_size=1, return_tensor=True)
                if x is None:
                    break
                x, y = x.float().to(self.device), y.float().to(self.device)
                pred = self.model(x)
                x = self._make_pred_mask_from_pred(pred)
                dice.append(float(DiceLoss.dice_loss(x, y, return_loss=False)))
                x_flat = x[:, 0].contiguous().view(-1)
                y_flat = y[:, 0].contiguous().view(-1)
                x_flat = x_flat.cpu()
                y_flat = y_flat.cpu()
                jaccard.append(jaccard_score(y_flat, x_flat))
            dataset.reset()

        avg_jaccard = sum(jaccard) / len(jaccard)
        avg_dice = sum(dice) / len(dice)
        dataset_dict["mini_cubes"]["jaccard_train"] = avg_jaccard
        dataset_dict["mini_cubes"]["dice_train"] = avg_dice

        if unused is False:
            self.metric_dict[dataset.x_data_dir[:-3]] = dataset_dict
        else:
            self.metric_dict_unused[dataset.x_data_dir[:-3]] = dataset_dict

    def _test_on_full_cubes(dataset):

        for key, value in dataset_map.items():
            if value == dataset.x_data_dir[:-2]:
                dataset_name = key
                break

        full_cubes_dir = dataset_full_cubes_map[dataset_name]
        full_cubes_labels_dir = dataset_full_cubes_labels_map[dataset_name]
        fcs = FullCubeSegmentator(
            model_path=os.path.join(self.config.model_path, "weights_sup.pt"),
            dataset_dir=full_cubes_dir,
            dataset_labels_dir=full_cubes_labels_dir,
            dataset_name=dataset_name,
        )

        metric_dict = fcs.compute_metrics_for_all_cubes()  # {"dice": .., "jaccard":}
        self.metric_dict[dataset.x_data_dir[:-3]]["full_cubes"] = metric_dict
        fcs.save_segmentation_examples()

    def _load_model(self, checkpoint_name: str):

        # NOT USED; CALLED TRAINER ISNTEAD

        checkpoint = torch.load(os.path.join(self.config.model_path_save, checkpoint_name), map_location=self.device)
        self.continue_to_testing = True

        if "ss" in checkpoint_name:
            completed_ss = checkpoint.get("completed_ss", False)
            if completed_ss is False:
                self.continue_to_testing = False
                # print("TRAINING NOT COMPLETED FOR {}. NOT TESTING".format(checkpoint_name))
            state_dict = checkpoint["model_state_dict_ss"]
        elif "sup" in checkpoint_name:
            completed_sup = checkpoint.get("completed_sup", False)
            if completed_sup is False:
                self.continue_to_testing = False
                # print("TRAINING NOT COMPLETED FOR {}. NOT TESTING".format(checkpoint_name))

            state_dict = checkpoint["model_state_dict_sup"]
        else:
            state_dict = checkpoint["state_dict"]

        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        self.model.load_state_dict(unParalled_state_dict)

    def _make_pred_mask_from_pred(self, pred, threshold=0.5):
        pred_mask_idxs = pred >= threshold
        pred_non_mask_idxs = pred < threshold
        pred[pred_mask_idxs] = float(1)
        pred[pred_non_mask_idxs] = float(0)
        return pred

    @staticmethod
    def iou(prediction, target_mask):
        intersection = np.logical_and(target_mask, prediction)
        union = np.logical_or(target_mask, prediction)
        iou_score = np.sum(intersection) / np.sum(union)


if __name__ == "__main__":

    config = load_object("objects/FROM_SCRATCH_cellari_heart_sup_2D_UNET_2D/only_supervised/run_1/config.pkl")
    dataset = load_object("objects/FROM_SCRATCH_cellari_heart_sup_2D_UNET_2D/only_supervised/run_1/dataset.pkl")
    dataset.x_data_dir = 
    t = Tester(config, dataset, test_all=False)
    t.test_segmentation()
