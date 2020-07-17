from dice_loss import DiceLoss
import numpy as np
from sklearn.metrics import jaccard_score
from dice_loss import DiceLoss
from unet3d import UNet3D
import json

from dataset import Dataset
from utils import get_unused_datasets, make_dir

from collections import defaultdict

import os
import torch

# TODO: check sup and ss has been completed flag


class Tester:
    def __init__(self, config, dataset, test_all=True):

        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_results_dir = os.path.join("test_results/", self.config.task_dir)
        make_dir(self.test_results_dir)
        self.test_all = test_all

        if self.config.model == "VNET_MG":
            self.model = UNet3D()

        self.model.to(self.device)

        self.model_weights_saved = os.listdir(self.config.model_path_save)
        self.metric_dict = dict()
        self.metric_dict_unused = dict()

    def test_segmentation(self):

        if isinstance(self.dataset, list):
            for dataset in self.dataset:
                self._test_dataset(dataset)
        else:
            self._test_dataset(self.dataset)

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

                with open(os.path.join(self.test_results_dir, "test_results_unused{}.json".format(file_nr)) as f:
                    json.dump(self.metric_dict_unused, f)

    def _test_dataset(self, dataset, unused=False):

        jaccard = []
        dice = []
        dataset_dict = dict()
        for model_file in self.model_weights_saved:
            dataset_dict.setdefault(model_file, {})
            self._load_model(model_file)
            if self.continue_to_testing is False:
                continue

            jaccard = []
            dice = []

            with torch.no_grad():
                self.model.eval()
                iteration = 0
                while True:
                    # if dataset.x_test_filenames:
                    #    print("USINGdataset.x_test_filenames)
                    x, y = dataset.get_test(batch_size=6, return_tensor=True)
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
                    iteration += 1
                dataset.reset()

            avg_jaccard = sum(jaccard) / len(jaccard)
            avg_dice = sum(dice) / len(dice)
            # print("AVG JACCARD ", str(avg_jaccard))
            # print("AVG DICE ", str(avg_dice))
            dataset_dict[model_file]["jaccard"] = avg_jaccard
            dataset_dict[model_file]["dice"] = avg_dice

        if unused is False:
            self.metric_dict[dataset.x_data_dir[:-3]] = dataset_dict
        else:
            self.metric_dict_unused[dataset.x_data_dir[:-3]] = dataset_dict

    def _load_model(self, checkpoint_name: str):

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
