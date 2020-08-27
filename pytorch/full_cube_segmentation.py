import os
from random import sample
import numpy as np
import SimpleITK as sitk
import torch
from copy import deepcopy
import json

# from skimage.util.shape import view_as_windows
import torch.nn.functional as F
import nibabel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import jaccard_score
import numpy as np

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

        self.trainer = Trainer(config=self.config, dataset=None)  # instanciating trainer to load and access model
        self.trainer.load_model(from_path=True, path=model_path, phase="sup")
        self.model = self.trainer.model

        self.all_cubes = [i for i in os.listdir(self.dataset_dir) if os.path.isfile(os.path.join(self.dataset_dir, i))]

    def get_all_cubes_which_were_used_for_training(self):

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

    def sample_k_full_cubes_which_were_used_for_training(self, k):
        corresponding_full_cubes = self.get_all_cubes_which_were_used_for_training()
        samp = sample(corresponding_full_cubes, k=k)
        return samp

    def sample_k_full_cubes_which_were_used_for_testing(self, k):
        corresponding_full_cubes = self.get_all_cubes_which_were_used_for_testing()
        samp = sample(corresponding_full_cubes, k=k)
        return samp

    def compute_metrics_for_all_cubes(self, inference_full_image=True):

        segmentations = []
        cubes_to_use = []
        full_cubes_used_for_testing = self.get_all_cubes_which_were_used_for_testing()
        full_cubes_used_for_training = self.get_all_cubes_which_were_used_for_training()
        cubes_to_use.extend(full_cubes_used_for_testing)
        cubes_to_use.extend(full_cubes_used_for_training)

        cubes_to_use_path = [os.path.join(self.dataset_dir, i) for i in cubes_to_use]
        label_cubes_of_cubes_to_use_path = [os.path.join(self.dataset_labels_dir, i) for i in cubes_to_use]

        metric_dict = dict()
        dice_test, dice_train, jaccard_test, jaccard_train = [], [], [], []

        for idx, cube_path in enumerate(cubes_to_use_path):
            np_array = self._load_cube_to_np_array(cube_path)  # (x,y,z)
            self.original_cube_dimensions = np_array.shape
            # np_array = self._normalize_cube(np_array, modality="mri")
            # patcher = Patcher(np_array, two_dim=self.two_dim)

            if inference_full_image is False:

                print("PATCH")
                patcher = Patcher(np_array, two_dim=self.two_dim)

                with torch.no_grad():
                    self.model.eval()
                    for patch_idx, patch in patcher:

                        patch = torch.unsqueeze(patch, 0)  # (1,C,H,W or 1) -> (1,1,C,H,W or 1)
                        if self.config.model.lower() in ("vnet_mg", "unet_3d", "unet_acs"):
                            patch, pad_tuple = pad_if_necessary_one_array(patch, return_pad_tuple=True)

                        pred = self.model(patch)
                        assert pred.shape == patch.shape, "{} vs {}".format(pred.shape, patch.shape)
                        # need to then unpad to reconstruct
                        if self.two_dim is True:
                            raise RuntimeError("SHOULD  NOT BE USED HERE")

                        pred = self._unpad_3d_array(pred, pad_tuple)
                        pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H,W) -> (1,C,H,W)
                        pred_mask = self._make_pred_mask_from_pred(pred)
                        patcher.predicitons_to_reconstruct_from[
                            :, patch_idx
                        ] = pred_mask  # update array in patcher that will construct full cube predicted mask
                pred_mask_full_cube = patcher.get_pred_mask_full_cube()
            else:

                full_cube_tensor = torch.Tensor(np_array)
                full_cube_tensor = torch.unsqueeze(full_cube_tensor, 0)  # (C,H,W) -> (1,C,H,W)
                full_cube_tensor = torch.unsqueeze(full_cube_tensor, 0)  # (1,C,H,W) -> (1,1,C,H,W)

                with torch.no_grad():
                    self.model.eval()
                    if self.two_dim is False:
                        if self.config.model.lower() in ("vnet_mg", "unet_3d", "unet_acs"):
                            full_cube_tensor, pad_tuple = pad_if_necessary_one_array(full_cube_tensor, return_pad_tuple=True)
                            try:
                                p = self.model(full_cube_tensor)
                                p.to("cpu")
                                pred = p
                                del p
                                dump_tensors()
                                torch.cuda.empty_cache()
                                dump_tensors()
                                torch.cuda.empty_cache()
                                pred = self._unpad_3d_array(pred, pad_tuple)
                                pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H,W) -> (1,C,H,W)
                                pred = torch.squeeze(pred, dim=0)
                                pred_mask_full_cube = self._make_pred_mask_from_pred(pred)
                                torch.cuda.empty_cache()
                                del pred

                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    print("TOO BIG FOR MEMORY, DEFAULTING TO PATCHING")
                                    # exit(0)
                                    res = self.compute_metrics_for_all_cubes(inference_full_image=False)
                                    return res

                    else:
                        pred_mask_full_cube = torch.zeros(self.original_cube_dimensions)
                        for z_idx in range(full_cube_tensor.size()[-1]):
                            tensor_slice = full_cube_tensor[..., z_idx]  # SLICE : (1,1,C,H,W) -> (1,1,C,H)
                            assert tensor_slice.shape == (1, 1, self.original_cube_dimensions[0], self.original_cube_dimensions[1])
                            pred = self.model(tensor_slice)
                            pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H) -> (1,C,H)
                            pred = torch.squeeze(pred, dim=0)  # (1,C,H) -> (C,H)
                            pred_mask_slice = self._make_pred_mask_from_pred(pred)
                            pred_mask_full_cube[..., z_idx] = pred_mask_slice

            # full_cube_segmentation_mask = patcher.get_pred_mask_full_cube()
            full_cube_label_tensor = torch.Tensor(self._load_cube_to_np_array(label_cubes_of_cubes_to_use_path[idx]))
            # full_cube_label_tensor = full_cube_label_tensor.to("cuda:0")
            pred_mask_full_cube = pred_mask_full_cube.to("cpu")
            dice_score = float(DiceLoss.dice_loss(pred_mask_full_cube, full_cube_label_tensor, return_loss=False))
            x_flat = pred_mask_full_cube.contiguous().view(-1)
            y_flat = full_cube_label_tensor.contiguous().view(-1)
            x_flat = x_flat.cpu()
            y_flat = y_flat.cpu()
            jac_score = jaccard_score(y_flat, x_flat)

            if idx < len(full_cubes_used_for_testing):
                dice_test.append(dice_score)
                jaccard_test.append(jac_score)
            else:
                dice_train.append(dice_score)
                jaccard_train.append(jac_score)

            print(idx)

        avg_jaccard_test = sum(jaccard_test) / len(jaccard_test)
        avg_jaccard_train = sum(jaccard_train) / len(jaccard_train)
        avg_dice_test = sum(dice_test) / len(dice_test)
        avg_dice_train = sum(dice_train) / len(dice_train)

        metric_dict["dice_test"] = avg_dice_test
        metric_dict["dice_train"] = avg_dice_train
        metric_dict["jaccard_test"] = avg_jaccard_test
        metric_dict["jaccard_train"] = avg_jaccard_train

        return metric_dict

    def save_segmentation_examples(self, nr_cubes=3, inference_full_image=True):

        # deal with recursion when defaulting to patchign
        segmentations = []
        cubes_to_use = []
        cubes_to_use.extend(self.sample_k_full_cubes_which_were_used_for_testing(nr_cubes))
        cubes_to_use.extend(self.sample_k_full_cubes_which_were_used_for_training(nr_cubes))

        cubes_to_use_path = [os.path.join(self.dataset_dir, i) for i in cubes_to_use]
        label_cubes_of_cubes_to_use_path = [os.path.join(self.dataset_labels_dir, i) for i in cubes_to_use]

        for cube_idx, cube_path in enumerate(cubes_to_use_path):
            np_array = self._load_cube_to_np_array(cube_path)  # (x,y,z)
            self.original_cube_dimensions = np_array.shape

            if inference_full_image is False:

                patcher = Patcher(np_array, two_dim=self.two_dim)

                with torch.no_grad():
                    self.model.eval()
                    for idx, patch in patcher:

                        patch = torch.unsqueeze(patch, 0)  # (1,C,H,W or 1) -> (1,1,C,H,W or 1)
                        if self.config.model.lower() in ("vnet_mg", "unet_3d", "unet_acs"):
                            patch, pad_tuple = pad_if_necessary_one_array(patch, return_pad_tuple=True)

                        pred = self.model(patch)
                        assert pred.shape == patch.shape, "{} vs {}".format(pred.shape, patch.shape)
                        # need to then unpad to reconstruct
                        if self.two_dim is True:
                            raise RuntimeError("SHOULD  NOT BE USED HERE")

                        pred = self._unpad_3d_array(pred, pad_tuple)
                        pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H,W) -> (1,C,H,W)
                        pred_mask = self._make_pred_mask_from_pred(pred)
                        patcher.predicitons_to_reconstruct_from[
                            :, idx
                        ] = pred_mask  # update array in patcher that will construct full cube predicted mask
                pred_mask_full_cube = patcher.get_pred_mask_full_cube()
                # segmentations.append(patcher.get_pred_mask_full_cube())
            else:

                full_cube_tensor = torch.Tensor(np_array)
                full_cube_tensor = torch.unsqueeze(full_cube_tensor, 0)  # (C,H,W) -> (1,C,H,W)
                full_cube_tensor = torch.unsqueeze(full_cube_tensor, 0)  # (1,C,H,W) -> (1,1,C,H,W)

                with torch.no_grad():
                    self.model.eval()
                    if self.two_dim is False:
                        if self.config.model.lower() in ("vnet_mg", "unet_3d", "unet_acs"):
                            full_cube_tensor, pad_tuple = pad_if_necessary_one_array(full_cube_tensor, return_pad_tuple=True)
                            try:
                                p = self.model(full_cube_tensor)
                                p.to("cpu")
                                pred = p
                                del p
                                dump_tensors()
                                torch.cuda.empty_cache()
                                dump_tensors()
                                torch.cuda.empty_cache()
                                pred = self._unpad_3d_array(pred, pad_tuple)
                                pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H,W) -> (1,C,H,W)
                                pred = torch.squeeze(pred, dim=0)
                                pred_mask_full_cube = self._make_pred_mask_from_pred(pred)
                                torch.cuda.empty_cache()
                                del pred

                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    print("TOO BIG FOR MEMORY, DEFAULTING TO PATCHING")
                                    # exit(0)
                                    self.save_segmentation_examples(inference_full_image=False)
                                    return

                            # segmentations.append(pred_mask_full_cube)
                    else:
                        pred_mask_full_cube = torch.zeros(self.original_cube_dimensions)
                        for z_idx in range(full_cube_tensor.size()[-1]):
                            tensor_slice = full_cube_tensor[..., z_idx]  # SLICE : (1,1,C,H,W) -> (1,1,C,H)
                            assert tensor_slice.shape == (1, 1, self.original_cube_dimensions[0], self.original_cube_dimensions[1])
                            pred = self.model(tensor_slice)
                            pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H) -> (1,C,H)
                            pred = torch.squeeze(pred, dim=0)  # (1,C,H) -> (C,H)
                            pred_mask_slice = self._make_pred_mask_from_pred(pred)
                            pred_mask_full_cube[..., z_idx] = pred_mask_slice

                        # segmentations.append(pred_mask_full_cube)

            # for idx, pred_mask_full_cube in enumerate(segmentations):

            print(cube_idx)

            if cube_idx < nr_cubes:
                save_dir = os.path.join(self.save_dir, self.dataset_name, "testing_examples/", cubes_to_use[cube_idx][:-4])
            else:
                save_dir = os.path.join(self.save_dir, self.dataset_name, "training_examples/", cubes_to_use[cube_idx][:-4])

            make_dir(save_dir)

            # save nii of segmentation
            pred_mask_full_cube = pred_mask_full_cube.cpu()
            nifty_img = nibabel.Nifti1Image(np.array(pred_mask_full_cube).astype(np.float32), np.eye(4))
            nibabel.save(nifty_img, os.path.join(save_dir, cubes_to_use[cube_idx][:-4] + ".nii.gz"))

            # self.save_3d_plot(np.array(pred_mask_full_cube), os.path.join(save_dir, "{}_plt3d.png".format(cubes_to_use[idx])))

            make_dir(os.path.join(save_dir, "slices/"))
            for z_idx in range(pred_mask_full_cube.shape[-1]):
                fig = plt.figure(figsize=(10, 5))
                plt.imshow(pred_mask_full_cube[:, :, z_idx], cmap=cm.Greys_r)
                fig.savefig(
                    os.path.join(save_dir, "slices/", "slice_{}.jpg".format(z_idx + 1)), bbox_inches="tight", dpi=150,
                )
                plt.close(fig=fig)

            label_tensor_of_cube = torch.Tensor(self._load_cube_to_np_array(label_cubes_of_cubes_to_use_path[cube_idx]))
            dice_score = float(DiceLoss.dice_loss(pred_mask_full_cube, label_tensor_of_cube, return_loss=False))
            x_flat = pred_mask_full_cube.contiguous().view(-1)
            y_flat = label_tensor_of_cube.contiguous().view(-1)
            x_flat = x_flat.cpu()
            y_flat = y_flat.cpu()
            jaccard_scr = jaccard_score(y_flat, x_flat)
            metrics = {"dice": dice_score, "jaccard": jaccard_scr}
            # print(dice)
            with open(os.path.join(save_dir, "dice.json"), "w") as f:
                json.dump(metrics, f)

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
            img_array = img_array.transpose(2, 1, 0)

        return img_array

    @staticmethod
    def _make_pred_mask_from_pred(pred, threshold=0.5):
        pred_mask_idxs = pred >= threshold
        pred_non_mask_idxs = pred < threshold
        pred[pred_mask_idxs] = float(1)
        pred[pred_non_mask_idxs] = float(0)
        return pred

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
