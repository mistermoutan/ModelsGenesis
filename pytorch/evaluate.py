from dice_loss import DiceLoss
import numpy as np
from sklearn.metrics import jaccard_score
from dice_loss import DiceLoss


class Tester:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def test_segmentation(self, test_dataset, use_val=False):

        # log with stats and tb_writer
        if hasattr(test_dataset, "datasets"):  # multiple datasets
            self.jaccard = []
            self.dice = []
            for dataset in test_dataset:
                jaccard = []
                dice = []
                while True:
                    x, y = dataset.get_test() if not use_val else dataset.get_val()
                    if x is None:
                        break
                    pred = self.model(x)
                    x = self._make_pred_mask_from_pred(pred)
                    x_flat = x[:, 0].contiguous().view(-1)
                    y_flat = y[:, 0].contiguous().view(-1)
                    jaccard.append(jaccard_score(y_flat, x_flat))
                    dice.append(float(DiceLoss.dice_loss(x_flat, y_flat, return_loss=False)))
                self.jaccard.append(sum(jaccard) / len(jaccard))
                self.dice.append(sum(dice) / len(dice))
        else:  # single dataset to test on
            while True:
                x, y = test_dataset.get_test() if not use_val else test_dataset.get_val()
                if x is None:
                    break
                pred = self.model(x)
                x = self._make_pred_mask_from_pred(pred)
                x_flat = x[:, 0].contiguous().view(-1)
                y_flat = y[:, 0].contiguous().view(-1)
                self.jaccard = jaccard_score(y_flat, x_flat)
                self.dice = float(DiceLoss.dice_loss(x_flat, y_flat, return_loss=False))

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
