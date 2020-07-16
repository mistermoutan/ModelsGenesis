from dice_loss import DiceLoss
import numpy as np
from sklearn.metrics import jaccard_score
from dice_loss import DiceLoss

from unet3d import UNet3D

import os
import torch


class Tester:
    def __init__(self, config, dataset):

        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config.model == "VNET_MG":
            self.model = UNet3D()

        self.model.to(self.device)

        self.model_weights_saved = os.listdir(self.config.model_path_save)
        self.metric_dict = dict()

    def test_segmentation(self):

        if isinstance(self.dataset, list):
            for dataset in self.dataset:
                self._test_dataset(dataset)
        else:
            self._test_dataset(dataset)

    def _load_model(self, checkpoint_name: str):

        checkpoint = torch.load(os.path.join(self.config.model_path_save, checkpoint_name), map_location=self.device)

        if "ss" in checkpoint_name:
            state_dict = checkpoint["model_state_dict_ss"]
        elif "sup" in checkpoint_name:
            state_dict = checkpoint["model_state_dict_sup"]
        else:
            state_dict = checkpoint["state_dict"]

        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        self.model.load_state_dict(unParalled_state_dict)

    def _test_dataset(self, dataset):

        self.jaccard = []
        self.dice = []

        for model_file in self.model_weights_saved:
            self._load_model(model_file)
            jaccard = []
            dice = []
            with torch.no_grad():
                self.model.eval()
                while True:
                    x, y = dataset.get_test(batch_size=6)
                    if x is None:
                        break
                    pred = self.model(x)
                    x = self._make_pred_mask_from_pred(pred)
                    x_flat = x[:, 0].contiguous().view(-1)
                    y_flat = y[:, 0].contiguous().view(-1)

                    jaccard.append(jaccard_score(y_flat, x_flat))
                    dice.append(float(DiceLoss.dice_loss(x_flat, y_flat, return_loss=False)))

            avg_jaccard = sum(jaccard) / len(jaccard)
            avg_dice = sum(dice) / len(dice)
            self.metric_dict[]

    def _make_pred_mask_from_pred(self, pred):
        pred_mask_idxs = pred >= self.config.threshold
        pred_non_mask_idxs = pred < self.config.threshold
        pred[pred_mask_idxs] = float(1)
        pred[pred_non_mask_idxs] = float(0)
        return pred

    @staticmethod
    def iou(prediction, target_mask):
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        iou_score = np.sum(intersection) / np.sum(union)
