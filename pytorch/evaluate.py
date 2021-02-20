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
import nibabel
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset import Dataset
from utils import *
from full_cube_segmentation import FullCubeSegmentator
from finetune import Trainer
from ACSConv.experiments.mylib.utils import categorical_to_one_hot


class Tester:
    def __init__(self, config, dataset, test_all: bool = False):

        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_results_dir = os.path.join("test_results/", self.config.task_dir)
        make_dir(self.test_results_dir)
        self.test_all = test_all
        model_path = os.path.join(self.config.model_path_save, "weights_sup.pt")
        self.task_dir = "/".join(i for i in model_path.split("/")[1:-1])
        self.save_dir = os.path.join("viz_samples/", self.task_dir)

        self.trainer = Trainer(config=self.config, dataset=None)  # instanciating trainer to load and access model
        self.trainer.load_model(
            from_path=True, path=os.path.join(self.config.model_path_save, "weights_sup.pt"), phase="sup", ensure_sup_is_completed=True
        )
        self.model = self.trainer.model

        self.metric_dict = dict()
        self.metric_dict_unused = dict()

        self.dataset_name = None
        for key, value in dataset_map.items():
            if value == dataset.x_data_dir[:-3]:
                self.dataset_name = key
                break
        assert self.dataset_name is not None, "Could not find dataset name key in dataset_map dictionary"

        self.print_yet = False

    def test_segmentation(self):

        full_cubes_datasets = ("task04_sup",)

        if isinstance(self.dataset, list):
            for dataset in self.dataset:
                self._test_dataset(dataset)
                if self.dataset_name.lower() in full_cubes_datasets:
                    self._test_on_full_cubes(dataset)
        else:
            self._test_dataset(self.dataset)
            if self.dataset_name.lower() in full_cubes_datasets:
                self._test_on_full_cubes(self.dataset)

        file_nr = 0
        filename = "test_results_both"
        while os.path.isfile(os.path.join(self.test_results_dir, "{}{}.json".format(filename, file_nr))):
            file_nr += 1

        # to check if testing remains consistent
        with open(os.path.join(self.test_results_dir, "{}{}.json".format(filename, file_nr)), "w") as f:
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
        dataset_dict.setdefault("mini_cubes", {})
        jaccard = []
        dice_binary = []
        dice_logits = []

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
                # batch size > 1 plus random shuffle of indexes in Dataset results in no having the exact same
                # testins result each time as batches are flattened and act "as one"
                # so you would be giving the metrics different tensors to work with
                x, y = dataset.get_val(batch_size=1, return_tensor=True)
                if x is None:
                    break
                x, y = x.float().to(self.device), y.float().to(self.device)
                if self.config.model.lower() in ("vnet_mg", "unet_3d", "unet_acs"):
                    x, pad_tuple = pad_if_necessary_one_array(x, return_pad_tuple=True)
                    pred = self.model(x)
                    pred = FullCubeSegmentator._unpad_3d_array(pred, pad_tuple)
                elif "fcn_resnet18" in self.config.model.lower():
                    x = torch.cat((x, x, x), dim=1)
                    if 86 in x.shape:
                        continue
                    pred = self.model(x)

                else:
                    pred = self.model(x)

                if "fcn_resnet18" not in self.config.model.lower():
                    dice_logits.append(float(DiceLoss.dice_loss(pred, y, return_loss=False)))
                else:
                    # match 2 channel output of network
                    y_one_hot = categorical_to_one_hot(y, dim=1, expand_dim=False)
                    dice_logits.append(float(DiceLoss.dice_loss(pred, y_one_hot, return_loss=False, skip_zero_sum=True)))

                pred = self._make_pred_mask_from_pred(pred)
                dice_binary.append(float(DiceLoss.dice_loss(pred, y, return_loss=False)))

                if x.shape[1] == 1:
                    # pred is binary here
                    x_flat = pred[:, 0].contiguous().view(-1)
                    y_flat = y[:, 0].contiguous().view(-1)
                    x_flat = x_flat.cpu()
                    y_flat = y_flat.cpu()
                    jaccard.append(jaccard_score(y_flat, x_flat))

                else:
                    # multi channel jaccard scenario
                    temp_jac = 0
                    for channel_idx in range(x.shape[1]):
                        x_flat = pred[:, channel_idx].contiguous().view(-1)
                        y_flat = y[:, channel_idx].contiguous().view(-1)
                        x_flat = x_flat.cpu()
                        y_flat = y_flat.cpu()
                        temp_jac += jaccard_score(y_flat, x_flat)
                    jaccard.append(temp_jac / x.shape[1])

            dataset.reset()

        avg_jaccard = sum(jaccard) / len(jaccard)
        avg_dice_soft = sum(dice_logits) / len(dice_logits)
        avg_dice_binary = sum(dice_binary) / len(dice_binary)
        dataset_dict["mini_cubes"]["jaccard_test"] = avg_jaccard
        dataset_dict["mini_cubes"]["dice_test_soft"] = avg_dice_soft
        dataset_dict["mini_cubes"]["dice_test_binary"] = avg_dice_binary

        jaccard = []
        dice_logits = []
        dice_binary = []
        with torch.no_grad():
            self.model.eval()
            while True:
                x, y = dataset.get_train(batch_size=1, return_tensor=True)
                if x is None:
                    break
                x, y = x.float().to(self.device), y.float().to(self.device)
                if self.config.model.lower() in ("vnet_mg", "unet_3d", "unet_acs"):
                    x, pad_tuple = pad_if_necessary_one_array(x, return_pad_tuple=True)
                    pred = self.model(x)
                    pred = FullCubeSegmentator._unpad_3d_array(pred, pad_tuple)
                elif "fcn_resnet18" in self.config.model.lower():
                    x = torch.cat((x, x, x), dim=1)
                    if 86 in x.shape:
                        continue
                    pred = self.model(x)
                else:
                    pred = self.model(x)

                if "fcn_resnet18" not in self.config.model.lower():
                    dice_logits.append(float(DiceLoss.dice_loss(pred, y, return_loss=False)))
                else:
                    # match 2 channel output of network
                    y_one_hot = categorical_to_one_hot(y, dim=1, expand_dim=False)
                    dice_logits.append(float(DiceLoss.dice_loss(pred, y_one_hot, return_loss=False, skip_zero_sum=True)))

                pred = self._make_pred_mask_from_pred(pred)
                dice_binary.append(float(DiceLoss.dice_loss(pred, y, return_loss=False)))

                if x.shape[1] == 1:
                    x_flat = pred[:, 0].contiguous().view(-1)
                    y_flat = y[:, 0].contiguous().view(-1)
                    x_flat = x_flat.cpu()
                    y_flat = y_flat.cpu()
                    jaccard.append(jaccard_score(y_flat, x_flat))

                else:
                    # multi channel jaccard scenario
                    temp_jac = 0
                    for channel_idx in range(x.shape[1]):
                        x_flat = pred[:, channel_idx].contiguous().view(-1)
                        y_flat = y[:, channel_idx].contiguous().view(-1)
                        x_flat = x_flat.cpu()
                        y_flat = y_flat.cpu()
                        temp_jac += jaccard_score(y_flat, x_flat)
                    jaccard.append(temp_jac / x.shape[1])

            dataset.reset()

        avg_jaccard = sum(jaccard) / len(jaccard)
        avg_dice_soft = sum(dice_logits) / len(dice_logits)
        avg_dice_binary = sum(dice_binary) / len(dice_binary)
        dataset_dict["mini_cubes"]["jaccard_train"] = avg_jaccard
        dataset_dict["mini_cubes"]["dice_train_soft"] = avg_dice_soft
        dataset_dict["mini_cubes"]["dice_train_binary"] = avg_dice_binary

        if unused is False:
            self.metric_dict[self.dataset_name] = dataset_dict
        else:
            self.metric_dict_unused[self.dataset_name] = dataset_dict

    def _test_on_full_cubes(self, dataset):

        full_cubes_dir = dataset_full_cubes_map[self.dataset_name]
        full_cubes_labels_dir = dataset_full_cubes_labels_map[self.dataset_name]
        fcs = FullCubeSegmentator(
            model_path=os.path.join(self.config.model_path_save, "weights_sup.pt"),
            dataset_dir=full_cubes_dir,
            dataset_labels_dir=full_cubes_labels_dir,
            dataset_name=self.dataset_name,
        )

        metric_dict = fcs.compute_metrics_for_all_cubes()  # {"dice": .., "jaccard":}
        self.metric_dict[self.dataset_name]["full_cubes"] = metric_dict
        fcs.save_segmentation_examples()

    def sample_k_mini_cubes_which_were_used_for_testing(self, k):

        # inside these cubes are N SAMPLES
        test_minicubes_filenames = self.dataset.x_val_filenames_original
        if self.dataset.x_test_filenames_original != []:
            test_minicubes_filenames.extend(self.dataset.x_test_filenames_original)
        test_minicubes_filenames.sort()
        if self.dataset_name == "lidc":
            assert len(test_minicubes_filenames) == 2
            return test_minicubes_filenames
        return test_minicubes_filenames[:k]

    def sample_k_mini_cubes_which_were_used_for_training(self, k):

        # inside these cubes are N SAMPLES
        train_minicubes_filenames = self.dataset.x_train_filenames_original
        train_minicubes_filenames.sort()
        if self.dataset_name == "lidc":
            assert len(train_minicubes_filenames) == 1
            return train_minicubes_filenames

        return train_minicubes_filenames[:k]
            state_dict = checkpoint["state_dict"]

        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        self.model.load_state_dict(unParalled_state_dict)

    def _make_pred_mask_from_pred(self, pred, threshold=0.5, print_=True):
        if pred.shape[1] == 1:
            if print_:
                if not self.print_yet:
                    print("1 CHANNEL OUTPUT FOR BINARY SEGMENTATION, USING THRESHOLD {} TO DETERMINE BINARY MASK".format(threshold))
                    self.print_yet = True
            pred_mask_idxs = pred >= threshold
            pred_non_mask_idxs = pred < threshold
            pred[pred_mask_idxs] = float(1)
            pred[pred_non_mask_idxs] = float(0)
        elif pred.shape[1] == 2:
            if print_:
                if not self.print_yet:
                    print("2 CHANNEL OUTPUT FOR BINARY SEGMENTATION, USING ARGMAX TO DETERMINE BINARY MASK")
                    self.print_yet = True
            pred = pred.argmax(1).unsqueeze(0)
        else:
            raise ValueError
        return pred

    @staticmethod
    def iou(prediction, target_mask):
        intersection = np.logical_and(target_mask, prediction)
        union = np.logical_or(target_mask, prediction)
        iou_score = np.sum(intersection) / np.sum(union)


if __name__ == "__main__":

    config = load_object("objects/FROM_SCRATCH_cellari_heart_UNET_3D/only_supervised/run_2/config.pkl")
    dataset = load_object("objects/FROM_SCRATCH_cellari_heart_UNET_3D/only_supervised/run_2/dataset.pkl")
    t = Tester(config, dataset, test_all=False)
    t.test_segmentation()

    # config = load_object("/home/moutan/ModelsGenesis/objects/FROM_SCRATCH_cellari_heart_sup_10_192_UNET_3D/only_supervised/run_1/config.pkl")
    # dataset = load_object("/home/moutan/ModelsGenesis/objects/FROM_SCRATCH_cellari_heart_sup_10_192_UNET_3D/only_supervised/run_1/dataset.pkl")
    # t = Tester(config, dataset, test_all=False)
    # t.test_segmentation()

    # config = load_object("/home/moutan/ModelsGenesis/objects/FROM_SCRATCH_cellari_heart_sup_10_192_UNET_ACS/only_supervised/run_1/config.pkl")
    # dataset = load_object("/home/moutan/ModelsGenesis/objects/FROM_SCRATCH_cellari_heart_sup_10_192_UNET_ACS/only_supervised/run_1/dataset.pkl")
    # t = Tester(config, dataset, test_all=False)
    # t.test_segmentation()

    # config = load_object("/home/moutan/ModelsGenesis/objects/FROM_SCRATCH_cellari_heart_sup_10_192_2D_UNET_2D/only_supervised/run_1/config.pkl")
    # dataset = load_object("/home/moutan/ModelsGenesis/objects/FROM_SCRATCH_cellari_heart_sup_10_192_UNET_2D/only_supervised/run_1/dataset.pkl")
    # t = Tester(config, dataset, test_all=False)
    # t.test_segmentation()
