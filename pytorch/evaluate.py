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

        full_cubes_datasets = ("task04_sup", "task01_sup", "cellari_heart_sup_10_192", "cellari_heart_sup")

        if isinstance(self.dataset, list):
            assert False, "SUPERVISION ON MULTIPLE DATASETS???"
            for dataset in self.dataset:
                self._test_dataset(dataset)
                self.save_segmentation_examples()
                if self.dataset_name.lower() in full_cubes_datasets:
                    self._test_on_full_cubes(dataset)
        else:
            self._test_dataset(self.dataset)
            self.save_segmentation_examples()
            if self.dataset_name.lower() in full_cubes_datasets:
                self._test_on_full_cubes(self.dataset)

        file_nr = 0
        filename = "test_results_new"
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
                if self.config.model.lower() in ("vnet_mg", "unet_3d", "unet_acs", "unet_acs_axis_aware_decoder", "unet_acs_with_cls"):
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

                if pred.shape[1] == 1:
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
                if self.config.model.lower() in ("vnet_mg", "unet_3d", "unet_acs", "unet_acs_axis_aware_decoder", "unet_acs_with_cls"):
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

                if pred.shape[1] == 1:
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

    def save_segmentation_examples(self, nr_cubes=5):

        # for mini cubes

        cubes_to_use = []
        cubes_to_use.extend(self.sample_k_mini_cubes_which_were_used_for_testing(nr_cubes))
        cubes_to_use.extend(self.sample_k_mini_cubes_which_were_used_for_training(nr_cubes))
        cubes_to_use_path = [os.path.join(self.dataset.x_data_dir, i) for i in cubes_to_use]
        label_cubes_of_cubes_to_use_path = []
        for cube_name in cubes_to_use:
            if os.path.isfile(os.path.join(self.dataset.y_data_dir, cube_name)):
                label_cubes_of_cubes_to_use_path.append(os.path.join(self.dataset.y_data_dir, cube_name))
            elif os.path.isfile(os.path.join(self.dataset.y_data_dir, cube_name[:-4] + "_target.npy")):
                label_cubes_of_cubes_to_use_path.append(os.path.join(self.dataset.y_data_dir, cube_name))
            else:
                raise FileNotFoundError

        self.two_dim = True if self.config.model.lower() == "unet_2d" else False

        for cube_idx, cube_path in enumerate(cubes_to_use_path):
            np_array = self._load_cube_to_np_array(cube_path)  # (N,x,y,z)
            np_array = (
                np_array[:3] if self.dataset_name != "lidc" else np_array[:10]
            )  # use number of samples lidc is different as training are all in the same file
            self.original_cube_dimensions = np_array.shape[1:]

            for sample_idx in range(np_array.shape[0]):
                mini_cube_tensor = torch.Tensor(np_array[sample_idx])  # (N,x,y,z) -> (x,y,z)
                mini_cube_tensor = torch.unsqueeze(mini_cube_tensor, 0)  # (x,y,z) -> (1,x,y,z)
                mini_cube_tensor = torch.unsqueeze(mini_cube_tensor, 0)  # (1,x,y,z) -> (1,1,x,y,z)
                with torch.no_grad():
                    self.trainer.model.eval()
                    if self.two_dim is False:
                        if self.config.model.lower() in (
                            "vnet_mg",
                            "unet_3d",
                            "unet_acs",
                            "unet_acs_axis_aware_decoder",
                            "unet_acs_with_cls",
                        ):
                            mini_cube_tensor, pad_tuple = pad_if_necessary_one_array(mini_cube_tensor, return_pad_tuple=True)
                            pred = self.model(mini_cube_tensor)
                            pred.to("cpu")
                            pred = FullCubeSegmentator._unpad_3d_array(pred, pad_tuple)
                            pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H,W) -> (1,C,H,W)
                            pred = torch.squeeze(pred, dim=0)
                            pred_mask_mini_cube = pred  # self._make_pred_mask_from_pred(pred)
                        elif "fcn_resnet" in self.config.model.lower():
                            x = torch.cat((mini_cube_tensor, mini_cube_tensor, mini_cube_tensor), dim=1)
                            if 86 in x.shape:
                                continue
                            pred_fcn = self.model(x)  # (1,2,C,H,W), channel 1 is target and channel 0 is background
                            pred_mask_mini_cube = pred_fcn[:, 1]  #  or pred[:, 1] - pred[:, 0] ; (1,1,C,H,W)
                            pred_mask_mini_cube = torch.squeeze(pred_mask_mini_cube, dim=0)  # (1, 1, C,H,W) -> (1,C,H,W)
                            pred_mask_mini_cube = torch.squeeze(pred_mask_mini_cube, dim=0)
                    else:
                        pred_mask_mini_cube = torch.zeros(self.original_cube_dimensions)
                        for z_idx in range(mini_cube_tensor.size()[-1]):
                            tensor_slice = mini_cube_tensor[..., z_idx]  # SLICE : (1,1,C,H,W) -> (1,1,C,H)
                            assert tensor_slice.shape == (1, 1, self.original_cube_dimensions[0], self.original_cube_dimensions[1])
                            pred = self.trainer.model(tensor_slice)
                            pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H) -> (1,C,H)
                            pred = torch.squeeze(pred, dim=0)  # (1,C,H) -> (C,H)
                            pred_mask_mini_cube[..., z_idx] = pred

                if cube_idx < nr_cubes:
                    save_dir = os.path.join(
                        self.save_dir,
                        self.dataset_name,
                        "testing_examples_mini/",
                        cubes_to_use[cube_idx][:-4] + "_sample{}".format(sample_idx),
                    )
                else:
                    save_dir = os.path.join(
                        self.save_dir,
                        self.dataset_name,
                        "training_examples_mini/",
                        cubes_to_use[cube_idx][:-4] + "_sample{}".format(sample_idx),
                    )

                make_dir(save_dir)

                pred_mask_mini_cube = pred_mask_mini_cube.cpu()  # logits mask
                if "fcn_resnet" not in self.config.model.lower():
                    pred_mask_mini_cube_binary = self._make_pred_mask_from_pred(pred_mask_mini_cube)  # binary mask
                else:
                    # we want to use argmax in the dual channel output when fcn resnet 18
                    # pred_mask_mini_cube is 1 channel and  "hacked it" in the case of fcn resnet 18 case
                    pred_mask_mini_cube_binary = self._make_pred_mask_from_pred(pred_fcn)
                    pred_mask_mini_cube_binary = torch.squeeze(pred_mask_mini_cube_binary, dim=0)
                    pred_mask_mini_cube_binary = torch.squeeze(pred_mask_mini_cube_binary, dim=0)
                    assert len(pred_mask_mini_cube_binary.shape) == 3

                # save nii's
                nifty_img = nibabel.Nifti1Image(np.array(pred_mask_mini_cube).astype(np.float32), np.eye(4))
                nibabel.save(
                    nifty_img, os.path.join(save_dir, cubes_to_use[cube_idx][:-4] + "_sample{}_logits_mask.nii.gz".format(sample_idx))
                )

                nifty_img = nibabel.Nifti1Image(np.array(pred_mask_mini_cube_binary).astype(np.float32), np.eye(4))
                nibabel.save(
                    nifty_img, os.path.join(save_dir, cubes_to_use[cube_idx][:-4] + "_sample{}_binary_mask.nii.gz".format(sample_idx))
                )

                # save .nii.gz of cube if is npy original full cube file
                if ".npy" in cube_path:
                    nifty_img = nibabel.Nifti1Image(np_array[sample_idx].astype(np.float32), np.eye(4))
                    nibabel.save(nifty_img, os.path.join(save_dir, cubes_to_use[cube_idx][:-4] + "_mini_cube{}.nii.gz".format(sample_idx)))

                # self.save_3d_plot(np.array(pred_mask_full_cube), os.path.join(save_dir, "{}_plt3d.png".format(cubes_to_use[idx]))))

                label_tensor_of_cube = torch.Tensor(self._load_cube_to_np_array(label_cubes_of_cubes_to_use_path[cube_idx]))[sample_idx]
                label_tensor_of_cube_masked = np.array(label_tensor_of_cube)
                label_tensor_of_cube_masked = np.ma.masked_where(
                    label_tensor_of_cube_masked < 0.5, label_tensor_of_cube_masked
                )  # it's binary anyway

                pred_mask_mini_cube_binary_masked = np.array(pred_mask_mini_cube_binary)
                pred_mask_mini_cube_binary_masked = np.ma.masked_where(
                    pred_mask_mini_cube_binary_masked < 0.5, pred_mask_mini_cube_binary_masked
                )  # it's binary anyway

                pred_mask_mini_cube_logits_masked = np.array(pred_mask_mini_cube)
                pred_mask_mini_cube_logits_masked = np.ma.masked_where(
                    pred_mask_mini_cube_logits_masked < 0.3, pred_mask_mini_cube_logits_masked
                )

                make_dir(os.path.join(save_dir, "slices_sample{}/".format(sample_idx)))

                for z_idx in range(pred_mask_mini_cube.shape[-1]):

                    # binary
                    fig = plt.figure(figsize=(10, 5))
                    plt.imshow(np_array[sample_idx][:, :, z_idx], cmap=cm.Greys_r)
                    plt.imshow(pred_mask_mini_cube_binary_masked[:, :, z_idx], cmap="Accent")
                    plt.axis("off")
                    fig.savefig(
                        os.path.join(save_dir, "slices_sample{}/".format(sample_idx), "slice_{}_binary.jpg".format(z_idx + 1)),
                        bbox_inches="tight",
                        dpi=150,
                    )
                    plt.close(fig=fig)

                    # logits
                    fig = plt.figure(figsize=(10, 5))
                    plt.imshow(np_array[sample_idx][:, :, z_idx], cmap=cm.Greys_r)
                    plt.imshow(pred_mask_mini_cube_logits_masked[:, :, z_idx], cmap="Blues", alpha=0.5)
                    plt.axis("off")
                    fig.savefig(
                        os.path.join(save_dir, "slices_sample{}/".format(sample_idx), "slice_{}_logits.jpg".format(z_idx + 1)),
                        bbox_inches="tight",
                        dpi=150,
                    )
                    plt.close(fig=fig)

                    # dist of logits histogram
                    distribution_logits = np.array(pred_mask_mini_cube[:, :, z_idx].contiguous().view(-1))
                    fig = plt.figure(figsize=(10, 5))
                    plt.hist(distribution_logits, bins=np.arange(min(distribution_logits), max(distribution_logits) + 0.05, 0.05))
                    fig.savefig(
                        os.path.join(save_dir, "slices_sample{}/".format(sample_idx), "slice_{}_logits_histogram.jpg".format(z_idx + 1)),
                        bbox_inches="tight",
                        dpi=150,
                    )
                    plt.close(fig=fig)

                    # save ground truth as wel, overlayed on original
                    fig = plt.figure(figsize=(10, 5))
                    plt.imshow(np_array[sample_idx][:, :, z_idx], cmap=cm.Greys_r)
                    plt.imshow(label_tensor_of_cube_masked[:, :, z_idx], cmap="jet")
                    plt.axis("off")
                    fig.savefig(
                        os.path.join(save_dir, "slices_sample{}/".format(sample_idx), "slice_{}_gt.jpg".format(z_idx + 1)),
                        bbox_inches="tight",
                        dpi=150,
                    )
                    plt.close(fig=fig)

                dice_score_soft = float(DiceLoss.dice_loss(pred_mask_mini_cube, label_tensor_of_cube, return_loss=False))
                dice_score_binary = float(DiceLoss.dice_loss(pred_mask_mini_cube_binary, label_tensor_of_cube, return_loss=False))
                x_flat = pred_mask_mini_cube_binary.contiguous().view(-1)
                y_flat = label_tensor_of_cube.contiguous().view(-1)
                x_flat = x_flat.cpu()
                y_flat = y_flat.cpu()
                jaccard_scr = jaccard_score(y_flat, x_flat)
                metrics = {"dice_logits": dice_score_soft, "dice_binary": dice_score_binary, "jaccard": jaccard_scr}
                # print(dice)
                with open(os.path.join(save_dir, "dice.json"), "w") as f:
                    json.dump(metrics, f)

    def _load_cube_to_np_array(self, cube_path):

        if ".npy" in cube_path:
            img_array = np.load(cube_path)
        else:
            # mini cubes were all made .npy in preprocessing
            raise ValueError
        return img_array

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
        elif len(pred.shape == 3):  # (just as x,y,z)
            pred_mask_idxs = pred >= threshold
            pred_non_mask_idxs = pred < threshold
            pred[pred_mask_idxs] = float(1)
            pred[pred_non_mask_idxs] = float(0)
        else:
            assert False, "got {}".format(pred.shape)
        return pred

    @staticmethod
    def iou(prediction, target_mask):
        intersection = np.logical_and(target_mask, prediction)
        union = np.logical_or(target_mask, prediction)
        iou_score = np.sum(intersection) / np.sum(union)


if __name__ == "__main__":

    config = load_object("objects/FROM_SCRATCH_fake_small_UNET_ACS/only_supervised/run_5/config.pkl")
    dataset = load_object("objects/FROM_SCRATCH_fake_small_UNET_ACS/only_supervised/run_5/dataset.pkl")
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
