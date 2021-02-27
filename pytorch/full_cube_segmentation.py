import os
from random import sample
import numpy as np
import SimpleITK as sitk
import torch
from copy import deepcopy
import json
from time import sleep

# from skimage.util.shape import view_as_windows
import torch.nn.functional as F
import nibabel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import jaccard_score
import numpy as np
from copy import deepcopy

from finetune import Trainer
from utils import *
from dice_loss import DiceLoss
from patcher import Patcher

# 3D Plotting
from skimage.measure import mesh_surface_area
from skimage.measure import marching_cubes_lewiner as marching_cubes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# from scipy.spatial import Delaunay
from scipy.ndimage.morphology import distance_transform_edt as dtrans


class FullCubeSegmentator:
    def __init__(self, model_path: str, dataset_dir: str, dataset_labels_dir: str, dataset_name: str):

        self.dataset_dir = dataset_dir  # original cubes, not extracted ones for training
        self.dataset_labels_dir = dataset_labels_dir
        self.task_dir = "/".join(
            i for i in model_path.split("/")[1:-1]
        )  # eg model_dir: #pretrained_weights/FROM_pretrained_weights/PRETRAIN_MG_FRAMEWORK_task01_ss_VNET_MG/run_1__task01_sup_VNET_MG/only_supervised/run_1/weights_sup.pt
        self.config = get_config_object_of_task_dir(self.task_dir)
        self.dataset = get_dataset_object_of_task_dir(self.task_dir)  # for knowing which were the training and testing cubes used
        self.two_dim = True if self.config.model.lower() == "unet_2d" else False
        self.dataset_name = dataset_name

        assert self.config is not None
        assert self.dataset is not None

        self.save_dir = os.path.join("viz_samples/", self.task_dir)
        make_dir(self.save_dir)

        self.model_path = model_path
        self.trainer = Trainer(config=self.config, dataset=None)  # instanciating trainer to load and access model
        # self.trainer.load_model(from_path=True, path=model_path, phase="sup")
        # self.model = self.trainer.model

        self.all_cubes = [i for i in os.listdir(self.dataset_dir) if os.path.isfile(os.path.join(self.dataset_dir, i))]

    def get_all_cubes_which_were_used_for_training(self):

        if self.dataset_name == "lidc":
            return ["{}.npy".format(i) for i in range(510)]
        train_minicubes_filenames = self.dataset.x_train_filenames_original
        corresponding_full_cubes = []
        for cube_name in self.all_cubes:
            list_of_files_corresponding_to_that_cube = [cube_name for s in train_minicubes_filenames if cube_name in s]
            assert len(list_of_files_corresponding_to_that_cube) in (0, 1), "There should only be 1 match or no match. {}".format(
                list_of_files_corresponding_to_that_cube
            )
            corresponding_full_cubes.extend(list_of_files_corresponding_to_that_cube)
        return corresponding_full_cubes

    def get_all_cubes_which_were_used_for_testing(self):

        if self.dataset_name == "lidc":
            return ["{}.npy".format(i) for i in range(510, 948)]

        test_mini_cube_file_names = self.dataset.x_val_filenames_original
        if self.dataset.x_test_filenames_original != []:
            test_mini_cube_file_names.extend(self.dataset.x_test_filenames_original)
        corresponding_full_cubes = []
        for cube_name in self.all_cubes:
            list_of_files_corresponding_to_that_cube = [cube_name for s in test_mini_cube_file_names if cube_name in s]
            assert len(list_of_files_corresponding_to_that_cube) in (0, 1), "There should only be 1 match or no match. {}".format(
                list_of_files_corresponding_to_that_cube
            )
            corresponding_full_cubes.extend(list_of_files_corresponding_to_that_cube)

        return corresponding_full_cubes

    def sample_k_full_cubes_which_were_used_for_training(self, k, deterministic=True):
        corresponding_full_cubes = self.get_all_cubes_which_were_used_for_training()
        if deterministic:
            corresponding_full_cubes.sort()
            samp = corresponding_full_cubes[:k]
        else:
            samp = sample(corresponding_full_cubes, k=k)
        return samp

    def sample_k_full_cubes_which_were_used_for_testing(self, k, deterministic=True):
        corresponding_full_cubes = self.get_all_cubes_which_were_used_for_testing()
        if deterministic:
            corresponding_full_cubes.sort()
            samp = corresponding_full_cubes[:k]
        else:
            samp = sample(corresponding_full_cubes, k=k)
        return samp

    def compute_metrics_for_all_cubes(self, inference_full_image=True):

        cubes_to_use = []

        dump_tensors()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        dump_tensors()
        torch.cuda.empty_cache()

        if "lidc" in self.dataset_name:
            return

        if hasattr(self.trainer, "model"):
            del self.trainer.model
            dump_tensors()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            dump_tensors()

        dump_tensors()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        dump_tensors()

        self.trainer.load_model(from_path=True, path=self.model_path, phase="sup", ensure_sup_is_completed=True)

        if inference_full_image is False:
            print("PATCHING Will be Done")

        full_cubes_used_for_testing = self.get_all_cubes_which_were_used_for_testing()
        full_cubes_used_for_training = self.get_all_cubes_which_were_used_for_training()
        cubes_to_use.extend(full_cubes_used_for_testing)
        cubes_to_use.extend(full_cubes_used_for_training)

        cubes_to_use_path = [os.path.join(self.dataset_dir, i) for i in cubes_to_use]
        label_cubes_of_cubes_to_use_path = [os.path.join(self.dataset_labels_dir, i) for i in cubes_to_use]

        metric_dict = dict()
        dice_logits_test, dice_logits_train, dice_binary_test, dice_binary_train, jaccard_test, jaccard_train = [], [], [], [], [], []

        for idx, cube_path in enumerate(cubes_to_use_path):
            np_array = self._load_cube_to_np_array(cube_path)  # (x,y,z)
            self.original_cube_dimensions = np_array.shape
            if sum([i for i in np_array.shape]) > 600 and self.two_dim is False:
                inference_full_image = False

            if self.dataset_name.lower() in ("task04_sup", "task01_sup", "cellari_heart_sup_10_192", "cellari_heart_sup"):
                inference_full_image = True

            if inference_full_image is False:
                print("CUBE TOO BIG, PATCHING")

                patcher = Patcher(np_array, two_dim=self.two_dim)

                with torch.no_grad():
                    self.trainer.model.eval()
                    for patch_idx, patch in patcher:

                        patch = torch.unsqueeze(patch, 0)  # (1,C,H,W or 1) -> (1,1,C,H,W or 1)
                        if self.config.model.lower() in (
                            "vnet_mg",
                            "unet_3d",
                            "unet_acs",
                            "unet_acs_axis_aware_decoder",
                            "unet_acs_with_cls",
                        ):
                            patch, pad_tuple = pad_if_necessary_one_array(patch, return_pad_tuple=True)

                        pred = self.trainer.model(patch)
                        assert pred.shape == patch.shape, "{} vs {}".format(pred.shape, patch.shape)
                        # need to then unpad to reconstruct
                        if self.two_dim is True:
                            raise RuntimeError("SHOULD  NOT BE USED HERE")

                        pred = self._unpad_3d_array(pred, pad_tuple)
                        pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H,W) -> (1,C,H,W)
                        # pred_mask = self._make_pred_mask_from_pred(pred)
                        patcher.predicitons_to_reconstruct_from[
                            :, patch_idx
                        ] = pred  # update array in patcher that will construct full cube predicted mask
                        del pred
                        dump_tensors()
                        torch.cuda.ipc_collect()
                        torch.cuda.empty_cache()
                        dump_tensors()

                pred_mask_full_cube = patcher.get_pred_mask_full_cube()

            else:

                full_cube_tensor = torch.Tensor(np_array)
                full_cube_tensor = torch.unsqueeze(full_cube_tensor, 0)  # (C,H,W) -> (1,C,H,W)
                full_cube_tensor = torch.unsqueeze(full_cube_tensor, 0)  # (1,C,H,W) -> (1,1,C,H,W)

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
                            full_cube_tensor, pad_tuple = pad_if_necessary_one_array(full_cube_tensor, return_pad_tuple=True)
                            try:
                                p = self.trainer.model(full_cube_tensor)
                                p.to("cpu")
                                pred = p
                                del p
                                dump_tensors()
                                torch.cuda.ipc_collect()
                                torch.cuda.empty_cache()
                                dump_tensors()
                                torch.cuda.empty_cache()
                                pred = self._unpad_3d_array(pred, pad_tuple)
                                pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H,W) -> (1,C,H,W)
                                pred = torch.squeeze(pred, dim=0)
                                pred_mask_full_cube = pred  # self._make_pred_mask_from_pred(pred)
                                torch.cuda.ipc_collect()
                                torch.cuda.empty_cache()
                                del pred

                            except RuntimeError as e:
                                if "out of memory" in str(e) or "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED" in str(e):
                                    print("TOO BIG FOR MEMORY, DEFAULTING TO PATCHING")
                                    # exit(0)
                                    dump_tensors()
                                    torch.cuda.ipc_collect()
                                    torch.cuda.empty_cache()
                                    dump_tensors()
                                    res = self.compute_metrics_for_all_cubes(inference_full_image=False)
                                    return res

                    else:
                        pred_mask_full_cube = torch.zeros(self.original_cube_dimensions)
                        for z_idx in range(full_cube_tensor.size()[-1]):
                            tensor_slice = full_cube_tensor[..., z_idx]  # SLICE : (1,1,C,H,W) -> (1,1,C,H)
                            assert tensor_slice.shape == (1, 1, self.original_cube_dimensions[0], self.original_cube_dimensions[1])
                            pred = self.trainer.model(tensor_slice)
                            pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H) -> (1,C,H)
                            pred = torch.squeeze(pred, dim=0)  # (1,C,H) -> (C,H)
                            pred_mask_slice = pred  # self._make_pred_mask_from_pred(pred)
                            pred_mask_full_cube[..., z_idx] = pred_mask_slice

            full_cube_label_tensor = torch.Tensor(self._load_cube_to_np_array(label_cubes_of_cubes_to_use_path[idx]))
            full_cube_label_tensor = self.adjust_label_cube_acording_to_dataset(full_cube_label_tensor)

            pred_mask_full_cube = pred_mask_full_cube.to("cpu")
            pred_mask_full_cube_binary = self._make_pred_mask_from_pred(pred_mask_full_cube)

            dice_score_soft = float(DiceLoss.dice_loss(pred_mask_full_cube, full_cube_label_tensor, return_loss=False))
            dice_score_binary = float(DiceLoss.dice_loss(pred_mask_full_cube_binary, full_cube_label_tensor, return_loss=False))

            x_flat = pred_mask_full_cube_binary.contiguous().view(-1)
            y_flat = full_cube_label_tensor.contiguous().view(-1)
            x_flat = x_flat.cpu()
            y_flat = y_flat.cpu()
            jac_score = jaccard_score(y_flat, x_flat)

            if idx < len(full_cubes_used_for_testing):
                dice_logits_test.append(dice_score_soft)
                dice_binary_test.append(dice_score_binary)
                jaccard_test.append(jac_score)
            else:
                dice_logits_train.append(dice_score_soft)
                dice_binary_train.append(dice_score_binary)
                jaccard_train.append(jac_score)

            dump_tensors()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            dump_tensors()

            print(idx)

        avg_jaccard_test = sum(jaccard_test) / len(jaccard_test)
        avg_jaccard_train = sum(jaccard_train) / len(jaccard_train)

        avg_dice_test_soft = sum(dice_logits_test) / len(dice_logits_test)
        avg_dice_test_binary = sum(dice_binary_test) / len(dice_binary_test)

        avg_dice_train_soft = sum(dice_logits_train) / len(dice_logits_train)
        avg_dice_train_binary = sum(dice_binary_train) / len(dice_binary_train)

        metric_dict["dice_test_soft"] = avg_dice_test_soft
        metric_dict["dice_test_binary"] = avg_dice_test_binary
        metric_dict["dice_train_soft"] = avg_dice_train_soft
        metric_dict["dice_train_binary"] = avg_dice_train_binary
        metric_dict["jaccard_test"] = avg_jaccard_test
        metric_dict["jaccard_train"] = avg_jaccard_train

        return metric_dict

    def save_segmentation_examples(self, nr_cubes=3, inference_full_image=True):

        # deal with recursion when defaulting to patchign

        if "lidc" in self.dataset_name:
            return

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        dump_tensors()

        if hasattr(self.trainer, "model"):
            del self.trainer.model
            dump_tensors()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            dump_tensors()

        if inference_full_image is False:
            print("PATCHING Will be Done")

        dump_tensors()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        dump_tensors()

        self.trainer.load_model(from_path=True, path=self.model_path, phase="sup", ensure_sup_is_completed=True)

        cubes_to_use = []
        cubes_to_use.extend(self.sample_k_full_cubes_which_were_used_for_testing(nr_cubes))
        cubes_to_use.extend(self.sample_k_full_cubes_which_were_used_for_training(nr_cubes))

        cubes_to_use_path = [os.path.join(self.dataset_dir, i) for i in cubes_to_use]
        label_cubes_of_cubes_to_use_path = [os.path.join(self.dataset_labels_dir, i) for i in cubes_to_use]

        for cube_idx, cube_path in enumerate(cubes_to_use_path):
            np_array = self._load_cube_to_np_array(cube_path)  # (x,y,z)
            self.original_cube_dimensions = np_array.shape
            if sum([i for i in np_array.shape]) > 500 and self.two_dim is False:
                print("CUBE TOO BIG, PATCHING")
                inference_full_image = False

            if inference_full_image is False:

                patcher = Patcher(np_array, two_dim=self.two_dim)

                with torch.no_grad():
                    self.trainer.model.eval()
                    for idx, patch in patcher:

                        patch = torch.unsqueeze(patch, 0)  # (1,C,H,W or 1) -> (1,1,C,H,W or 1)
                        if self.config.model.lower() in (
                            "vnet_mg",
                            "unet_3d",
                            "unet_acs",
                            "unet_acs_axis_aware_decoder",
                            "unet_acs_with_cls",
                        ):
                            patch, pad_tuple = pad_if_necessary_one_array(patch, return_pad_tuple=True)

                        pred = self.trainer.model(patch)
                        assert pred.shape == patch.shape, "{} vs {}".format(pred.shape, patch.shape)
                        # need to then unpad to reconstruct
                        if self.two_dim is True:
                            raise RuntimeError("SHOULD  NOT BE USED HERE")

                        pred = self._unpad_3d_array(pred, pad_tuple)
                        pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H,W) -> (1,C,H,W)
                        pred_mask = pred  # self._make_pred_mask_from_pred(pred)
                        del pred

                        patcher.predicitons_to_reconstruct_from[
                            :, idx
                        ] = pred_mask  # update array in patcher that will construct full cube predicted mask

                        dump_tensors()
                        torch.cuda.ipc_collect()
                        torch.cuda.empty_cache()
                        dump_tensors()

                pred_mask_full_cube = patcher.get_pred_mask_full_cube()
                # segmentations.append(patcher.get_pred_mask_full_cube())
            else:

                full_cube_tensor = torch.Tensor(np_array)
                full_cube_tensor = torch.unsqueeze(full_cube_tensor, 0)  # (C,H,W) -> (1,C,H,W)
                full_cube_tensor = torch.unsqueeze(full_cube_tensor, 0)  # (1,C,H,W) -> (1,1,C,H,W)

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
                            full_cube_tensor, pad_tuple = pad_if_necessary_one_array(full_cube_tensor, return_pad_tuple=True)
                            try:
                                p = self.trainer.model(full_cube_tensor)
                                p.to("cpu")
                                pred = p
                                del p
                                dump_tensors()
                                torch.cuda.ipc_collect()
                                torch.cuda.empty_cache()
                                dump_tensors()
                                torch.cuda.empty_cache()
                                pred = self._unpad_3d_array(pred, pad_tuple)
                                pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H,W) -> (1,C,H,W)
                                pred = torch.squeeze(pred, dim=0)
                                pred_mask_full_cube = pred  # self._make_pred_mask_from_pred(pred)
                                torch.cuda.ipc_collect()
                                torch.cuda.empty_cache()
                                del pred

                            except RuntimeError as e:
                                if "out of memory" in str(e) or "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED" in str(e):
                                    print("TOO BIG FOR MEMORY, DEFAULTING TO PATCHING")
                                    # exit(0)
                                    dump_tensors()
                                    torch.cuda.ipc_collect()
                                    torch.cuda.empty_cache()
                                    dump_tensors()
                                    self.save_segmentation_examples(inference_full_image=False)
                                    return

                            # segmentations.append(pred_mask_full_cube)
                    else:
                        pred_mask_full_cube = torch.zeros(self.original_cube_dimensions)
                        for z_idx in range(full_cube_tensor.size()[-1]):
                            tensor_slice = full_cube_tensor[..., z_idx]  # SLICE : (1,1,C,H,W) -> (1,1,C,H)
                            assert tensor_slice.shape == (1, 1, self.original_cube_dimensions[0], self.original_cube_dimensions[1])
                            pred = self.trainer.model(tensor_slice)
                            pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H) -> (1,C,H)
                            pred = torch.squeeze(pred, dim=0)  # (1,C,H) -> (C,H)
                            pred_mask_slice = pred  # self._make_pred_mask_from_pred(pred)
                            pred_mask_full_cube[..., z_idx] = pred_mask_slice

                        # segmentations.append(pred_mask_full_cube)

            # for idx, pred_mask_full_cube in enumerate(segmentations):

            print(cube_idx)

            if cube_idx < nr_cubes:
                if inference_full_image is True:
                    save_dir = os.path.join(self.save_dir, self.dataset_name, "testing_examples_full/", cubes_to_use[cube_idx][:-4])
                else:
                    save_dir = os.path.join(
                        self.save_dir, self.dataset_name, "testing_examples_full/", cubes_to_use[cube_idx][:-4] + "_with_patcher"
                    )
            else:
                if inference_full_image is True:
                    save_dir = os.path.join(self.save_dir, self.dataset_name, "training_examples_full/", cubes_to_use[cube_idx][:-4])
                else:
                    save_dir = os.path.join(
                        self.save_dir, self.dataset_name, "training_examples_full/", cubes_to_use[cube_idx][:-4] + "_with_patcher"
                    )

            make_dir(save_dir)

            # save nii of segmentation
            pred_mask_full_cube = pred_mask_full_cube.cpu()  # logits mask
            pred_mask_full_cube_binary = self._make_pred_mask_from_pred(pred_mask_full_cube)  # binary mask

            nifty_img = nibabel.Nifti1Image(np.array(pred_mask_full_cube).astype(np.float32), np.eye(4))
            nibabel.save(nifty_img, os.path.join(save_dir, cubes_to_use[cube_idx][:-4] + "_logits_mask.nii.gz"))

            nifty_img = nibabel.Nifti1Image(np.array(pred_mask_full_cube_binary).astype(np.float32), np.eye(4))
            nibabel.save(nifty_img, os.path.join(save_dir, cubes_to_use[cube_idx][:-4] + "_binary_mask.nii.gz"))

            # save .nii.gz of cube if is npy original full cube file
            if ".npy" in cube_path:
                nifty_img = nibabel.Nifti1Image(np_array.astype(np.float32), np.eye(4))
                nibabel.save(nifty_img, os.path.join(save_dir, cubes_to_use[cube_idx][:-4] + "_cube.nii.gz"))

            # self.save_3d_plot(np.array(pred_mask_full_cube), os.path.join(save_dir, "{}_plt3d.png".format(cubes_to_use[idx])))

            label_tensor_of_cube = torch.Tensor(self._load_cube_to_np_array(label_cubes_of_cubes_to_use_path[cube_idx]))
            label_tensor_of_cube = self.adjust_label_cube_acording_to_dataset(label_tensor_of_cube)
            label_tensor_of_cube_masked = np.array(label_tensor_of_cube)
            label_tensor_of_cube_masked = np.ma.masked_where(
                label_tensor_of_cube_masked < 0.5, label_tensor_of_cube_masked
            )  # it's binary anyway

            pred_mask_full_cube_binary_masked = np.array(pred_mask_full_cube_binary)
            pred_mask_full_cube_binary_masked = np.ma.masked_where(
                pred_mask_full_cube_binary_masked < 0.5, pred_mask_full_cube_binary_masked
            )  # it's binary anyway

            pred_mask_full_cube_logits_masked = np.array(pred_mask_full_cube)
            pred_mask_full_cube_logits_masked = np.ma.masked_where(
                pred_mask_full_cube_logits_masked < 0.3, pred_mask_full_cube_logits_masked
            )  # it's binary anyway

            make_dir(os.path.join(save_dir, "slices/"))

            for z_idx in range(pred_mask_full_cube.shape[-1]):

                # binary
                fig = plt.figure(figsize=(10, 5))
                plt.imshow(np_array[:, :, z_idx], cmap=cm.Greys_r)
                plt.imshow(pred_mask_full_cube_binary_masked[:, :, z_idx], cmap="Accent")
                plt.axis("off")
                fig.savefig(
                    os.path.join(save_dir, "slices/", "slice_{}_binary.jpg".format(z_idx + 1)),
                    bbox_inches="tight",
                    dpi=150,
                )
                plt.close(fig=fig)

                # logits
                fig = plt.figure(figsize=(10, 5))
                plt.imshow(np_array[:, :, z_idx], cmap=cm.Greys_r)
                plt.imshow(pred_mask_full_cube_logits_masked[:, :, z_idx], cmap="Blues", alpha=0.5)
                plt.axis("off")
                fig.savefig(
                    os.path.join(save_dir, "slices/", "slice_{}_logits.jpg".format(z_idx + 1)),
                    bbox_inches="tight",
                    dpi=150,
                )
                plt.close(fig=fig)

                # dist of logits histogram
                distribution_logits = np.array(pred_mask_full_cube[:, :, z_idx].contiguous().view(-1))
                fig = plt.figure(figsize=(10, 5))
                plt.hist(distribution_logits, bins=np.arange(min(distribution_logits), max(distribution_logits) + 0.05, 0.05))
                fig.savefig(
                    os.path.join(save_dir, "slices/", "slice_{}_logits_histogram.jpg".format(z_idx + 1)),
                    bbox_inches="tight",
                    dpi=150,
                )
                plt.close(fig=fig)

                # save ground truth as wel, overlayed on original
                fig = plt.figure(figsize=(10, 5))
                plt.imshow(np_array[:, :, z_idx], cmap=cm.Greys_r)
                plt.imshow(label_tensor_of_cube_masked[:, :, z_idx], cmap="jet")
                plt.axis("off")
                fig.savefig(
                    os.path.join(save_dir, "slices/", "slice_{}_gt.jpg".format(z_idx + 1)),
                    bbox_inches="tight",
                    dpi=150,
                )
                plt.close(fig=fig)

            dice_score_soft = float(DiceLoss.dice_loss(pred_mask_full_cube, label_tensor_of_cube, return_loss=False))
            dice_score_binary = float(DiceLoss.dice_loss(pred_mask_full_cube_binary, label_tensor_of_cube, return_loss=False))
            x_flat = pred_mask_full_cube_binary.contiguous().view(-1)
            y_flat = pred_mask_full_cube_binary.contiguous().view(-1)
            x_flat = x_flat.cpu()
            y_flat = y_flat.cpu()
            jaccard_scr = jaccard_score(y_flat, x_flat)
            metrics = {"dice_logits": dice_score_soft, "dice_binary": dice_score_binary, "jaccard": jaccard_scr}
            # print(dice)
            with open(os.path.join(save_dir, "dice.json"), "w") as f:
                json.dump(metrics, f)

            dump_tensors()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            dump_tensors()
            dump_tensors()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            dump_tensors()

    def adjust_label_cube_acording_to_dataset(self, label_cube):

        if "Task01_BrainTumour" in dataset_map[self.dataset_name]:
            above_0_idxs = label_cube >= 1
            label_cube[above_0_idxs] = float(1)
        elif "Task03_Liver" in dataset_map[self.dataset_name]:
            # use mask for liver only
            non_liver_idxs = label_cube != 1
            label_cube[non_liver_idxs] = float(0)
        elif "Task04_Hippocampus" in dataset_map[self.dataset_name]:
            strucutures_idxs = label_cube >= 1
            label_cube[strucutures_idxs] = float(1)
        elif "Task05_Prostate" in dataset_map[self.dataset_name]:
            strucutures_idxs = label_cube >= 1
            label_cube[strucutures_idxs] = float(1)
        elif "Task07_Pancreas" in dataset_map[self.dataset_name]:
            # use mask for pancreas only
            non_pancreas_idxs = label_cube != 1
            label_cube[non_pancreas_idxs] = float(0)
        elif "Task08_HepaticVessel" in dataset_map[self.dataset_name]:
            # use mask for vessel only
            non_vessel_idxs = label_cube != 1
            label_cube[non_vessel_idxs] = float(0)

        return label_cube

    @staticmethod
    def _normalize_cube(np_array, modality="mri"):
        if modality == "ct":
            hu_max, hu_min = 1000, -1000
        if modality == "mri":
            hu_max, hu_min = 4000, 0

        # min-max normalization
        while np.max(np_array) < hu_max:
            hu_max -= 10
        while np.min(np_array) > hu_min:
            hu_min += 10

        if hu_max != 0 and hu_min != 0:
            np_array[np_array < hu_min] = hu_min
            np_array[np_array > hu_max] = hu_max
            np_array = 1.0 * (np_array - hu_min) / (hu_max - hu_min)

        return np_array

    @staticmethod
    def _unpad_3d_array(tensor, pad_tuple):
        # if pad(2,3) then [2:x-3]
        assert len(pad_tuple) == 2 * len(tensor.size()), "{} | {}".format(len(pad_tuple), tensor.size())
        pt = pad_tuple
        shape = tensor.size()
        tensor = tensor[:, :, pt[4] : shape[2] - pt[5], pt[2] : shape[3] - pt[3], pt[0] : shape[4] - pt[1]]
        return tensor

    def _load_cube_to_np_array(self, cube_path):

        if ".npy" in cube_path:
            img_array = np.load(cube_path)
        else:
            itk_img = sitk.ReadImage(cube_path)
            img_array = sitk.GetArrayFromImage(itk_img)

            if len(img_array.shape) == 4:
                if "Task01_BrainTumour" in dataset_map[self.dataset_name]:
                    img_array = img_array[1]
                elif "Task05_Prostate" in dataset_map[self.dataset_name]:
                    img_array = img_array[0]
                else:
                    raise ValueError("We should be working with 1 channel only")

            img_array = img_array.transpose(2, 1, 0)

        return img_array

    @staticmethod
    def _make_pred_mask_from_pred(pred, threshold=0.5):
        res = deepcopy(pred)
        pred_mask_idxs = res >= threshold
        pred_non_mask_idxs = res < threshold
        res[pred_mask_idxs] = float(1)
        res[pred_non_mask_idxs] = float(0)
        return res

    def save_3d_plot(self, mask_array, save_dir, figsize=(15, 10), step=1, edgecolor="0.2", cmap="viridis", backend="matplotlib"):

        # TODO: fix this shit
        verts, faces, _, _ = marching_cubes(mask_array.astype(np.float), 0.5, spacing=(1, 1, 1), step_size=step)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        t = np.linspace(0, 1, faces.shape[0])
        mesh = Poly3DCollection(verts[faces], edgecolor=edgecolor, facecolors=plt.cm.cmap_d[cmap](t))
        ax.add_collection3d(mesh)

        # ceil = max(self.bbox_dims(pad=[(1, 1), (1, 1), (1, 1)]))
        # ceil = int(np.round(ceil))

        # ax.set_xlim(0, ceil)
        # ax.set_xlabel("length (mm)")

        # ax.set_ylim(0, ceil)
        # ax.set_ylabel("length (mm)")

        # ax.set_zlim(0, ceil)
        # ax.set_zlabel("length (mm)")

        plt.tight_layout()
        fig.savefig(save_dir, bbox_inches="tight", dpi=150)
        plt.close(fig=fig)


if __name__ == "__main__":

    f = FullCubeSegmentator(
        model_path="pretrained_weights/FROM_SCRATCH_cellari_heart_sup_10_192_UNET_3D/only_supervised/run_2/weights_sup.pt",
        dataset_dir="pytorch/datasets/heart_mri/datasets/x_cubes_full",
        dataset_labels_dir="pytorch/datasets/heart_mri/datasets/y_cubes_full",
        dataset_name="heart_cellari",
    )
    # f.compute_metrics_for_all_cubes()
    # metric_dict = f.compute_metrics_for_all_cubes()
    # print(metric_dict)
    f.save_segmentation_examples()
    # metric_dict_2 = f.compute_metrics_for_all_cubes(inference_full_image=False)
    # print(metric_dict_2)
    # f.save_segmentation_examples()
    # from utils import pad_if_necessary_one_array

    # a = torch.randn(1, 1, 64, 64, 12)
    # b, pt = pad_if_necessary_one_array(a, return_pad_tuple=True)
    # unp = FullCubeSegmentationConstructor._unpad_3d_array

    # c = unp(b, pt)
    # print((c == a).all())
