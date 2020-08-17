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

from finetune import Trainer
from utils import *
from dice_loss import DiceLoss

# 3D Plotting
from skimage.measure import mesh_surface_area
from skimage.measure import marching_cubes_lewiner as marching_cubes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# from scipy.spatial import Delaunay
from scipy.ndimage.morphology import distance_transform_edt as dtrans


class FullCubeSegmentationVisualizer:
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

    def sample_k_full_cubes_which_were_used_for_training(self, k):
        train_minicubes_filenames = self.dataset.x_train_filenames_original
        corresponding_full_cubes = []
        for cube_name in self.all_cubes:
            list_of_files_corresponding_to_that_cube = [s for s in train_minicubes_filenames if cube_name in s]
            assert len(list_of_files_corresponding_to_that_cube) in (0, 1), "There should only be 1 match or no match. {}".format(
                list_of_files_corresponding_to_that_cube
            )
            corresponding_full_cubes.extend(list_of_files_corresponding_to_that_cube)
        samp = sample(corresponding_full_cubes, k=k)
        return samp

    def sample_k_full_cubes_which_were_used_for_testing(self, k):
        test_mini_cube_file_names = self.dataset.x_val_filenames_original
        if self.dataset.x_test_filenames_original != []:
            test_mini_cube_file_names.extend(self.dataset.x_test_filenames_original)
        corresponding_full_cubes = []
        for cube_name in self.all_cubes:
            list_of_files_corresponding_to_that_cube = [s for s in test_mini_cube_file_names if cube_name in s]
            assert len(list_of_files_corresponding_to_that_cube) in (0, 1), "There should only be 1 match or no match. {}".format(
                list_of_files_corresponding_to_that_cube
            )
            corresponding_full_cubes.extend(list_of_files_corresponding_to_that_cube)
        samp = sample(corresponding_full_cubes, k=k)
        return samp

    def get_segmentation_examples(self, nr_cubes=3, save_slices=True):

        segmentations = []
        self.cubes_to_use = []
        self.cubes_to_use.extend(self.sample_k_full_cubes_which_were_used_for_testing(nr_cubes))
        self.cubes_to_use.extend(self.sample_k_full_cubes_which_were_used_for_training(nr_cubes))
        self.cubes_to_use_path = [os.path.join(self.dataset_dir, i) for i in self.cubes_to_use]
        self.label_cubes_of_cubes_to_use_path = [os.path.join(self.dataset_labels_dir, i) for i in self.cubes_to_use]

        for cube_path in self.cubes_to_use_path:
            np_array = self._load_cube_to_np_array(cube_path)  # (x,y,z)
            patcher = Patcher(np_array, two_dim=self.two_dim)
            with torch.no_grad():
                self.model.eval()
                for idx, patch in patcher:
                    patch = torch.unsqueeze(patch, 0)  # (1,C,H,W) -> (1,1,C,H,W)
                    if self.config.model.lower() in ("vnet_mg", "unet_3d", "unet_acs"):
                        patch, pad_tuple = pad_if_necessary_one_array(patch, return_pad_tuple=True)
                    if self.two_dim is True:
                        pad_tuple = tuple([0 for i in range(len(patch.shape) * 2)])
                        patch = patch.squeeze(dim=-1)

                    pred = self.model(patch)
                    # need to then unpad to reconstruct
                    if self.two_dim is True:
                        pred = pred.unsqueeze(dim=-1)

                    pred = self._unpad_3d_array(pred, pad_tuple)
                    pred = torch.squeeze(pred, dim=0)  # (1, 1, C,H,W) -> (1,C,H,W)
                    pred_mask = self._make_pred_mask_from_pred(pred)
                    patcher.predicitons_to_reconstruct_from[
                        :, idx
                    ] = pred_mask  # update array in patcher that will construct full cube predicted mask
            segmentations.append(patcher.get_pred_mask_full_cube())

        for idx, seg in enumerate(segmentations):
            # torch.save(seg, os.path.join(self.save_dir, self.cubes_to_use_path[idx]))
            if idx < nr_cubes:
                save_dir = os.path.join(self.save_dir, self.dataset_name, "testing_examples/", self.cubes_to_use[idx][:-4])
            else:
                save_dir = os.path.join(self.save_dir, self.dataset_name, "training_examples/", self.cubes_to_use[idx][:-4])

            make_dir(save_dir)

            # save nii of segmentation
            nifty_img = nibabel.Nifti1Image(np.array(seg).astype(np.float32), np.eye(4))
            nibabel.save(nifty_img, os.path.join(save_dir, self.cubes_to_use[idx][:-4] + ".nii.gz"))

            # self.save_3d_plot(np.array(seg), os.path.join(save_dir, "{}_plt3d.png".format(self.cubes_to_use[idx])))

            if save_slices is True:
                make_dir(os.path.join(save_dir, "slices/"))
                for z_idx in range(seg.shape[-1]):
                    fig = plt.figure(figsize=(10, 5))
                    plt.imshow(seg[:, :, z_idx], cmap=cm.Greys_r)
                    fig.savefig(
                        os.path.join(save_dir, "slices/", "slice_{}.jpg".format(z_idx + 1)), bbox_inches="tight", dpi=150,
                    )

            label_tensor_of_cube = torch.Tensor(self._load_cube_to_np_array(self.label_cubes_of_cubes_to_use_path[idx]))
            dice = {"dice": float(DiceLoss.dice_loss(seg, label_tensor_of_cube, return_loss=False))}
            with open(os.path.join(save_dir, "dice.json"), "w") as f:
                json.dump(dice, f)

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


class Patcher:

    # https://discuss.pytorch.org/t/patch-making-does-pytorch-have-anything-to-offer/33850/10

    def __init__(self, cube: np.ndarray or torch.Tensor, two_dim=False):

        self.cube = cube
        self.two_dim = two_dim
        self._build_patches()

    def __iter__(self):

        for idx in range(self.patches.size()[1]):
            yield idx, self.patches[:, idx]  # (1, C, H, W)

    def _build_patches(self):

        self.original_cube_dimensions = self.cube.shape
        print("CUBE DIMENSIONS: {}".format(self.original_cube_dimensions))

        if self.two_dim is False:
            self.kernel_size = [64, 64, 32]  # kernel size
            stride = [64, 64, 32]  # stride
        else:
            # patches for two dimensional model
            self.kernel_size = [64, 64, 1]
            stride = [64, 64, 1]
            for idx in range(len(self.kernel_size)):
                if self.kernel_size[idx] > self.original_cube_dimensions[idx]:
                    self.kernel_size[idx] = self.original_cube_dimensions[idx]
                    stride[idx] = self.kernel_size[idx]

        self.cube = torch.Tensor(self.cube)
        self._pad_to_cover_all()
        print("CUBE DIMENSIONS POST PADDING TO COVER ALL: {}".format(self.cube.shape))
        self.cube = torch.unsqueeze(self.cube, 0)  # (x, y, z) -> (1,x,y,z)
        assert self.cube.size()[0] == 1

        self.patches = (
            self.cube.unfold(1, self.kernel_size[0], stride[0])
            .unfold(2, self.kernel_size[1], stride[1])
            .unfold(3, self.kernel_size[2], stride[2])
        )

        self.unfold_shape = self.patches.size()

        self.patches = self.patches.contiguous().view(
            self.patches.size(0), -1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]
        )  # (1, Number patches, kernel_size[0],kernel_size[1]. kernel_size[2])
        print("PATCH TENSOR DIMENSION: {}".format(self.patches.size()))

        # init tensor that is to be updsted with model predictions
        self.predicitons_to_reconstruct_from = torch.zeros(self.patches.size())

    def get_pred_mask_full_cube(self):

        patches_orig = self.predicitons_to_reconstruct_from.view(self.unfold_shape)  # torch.Size([1, 8, 8, 1, 64, 64, 12])
        self.output_c = self.unfold_shape[1] * self.unfold_shape[4]
        self.output_h = self.unfold_shape[2] * self.unfold_shape[5]
        self.output_w = self.unfold_shape[3] * self.unfold_shape[6]
        patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()  # torch.Size([1, 8, 64, 8, 64, 1, 12])
        patches_orig = patches_orig.view(1, self.output_c, self.output_h, self.output_w)  # torch.Size of padded cube (1, C, H, W)
        patches_orig = torch.squeeze(patches_orig)  # (x,y,z)
        patches_orig = self._unpad(patches_orig)
        assert patches_orig.size() == self.original_cube_dimensions, "{} != {}".format(patches_orig.size(), self.original_cube_dimensions)
        return patches_orig

        # Check for equality
        # print((patches_orig == self.cube[:, : self.output_c, : self.output_h, : self.output_w]).all())

    def _unpad(self, tensor):

        # (x,y,z)
        if hasattr(self, "pad_tuple"):
            assert len(self.pad_tuple) == 2 * len(tensor.size())
            pt = self.pad_tuple
            shape = tensor.size()
            tensor = tensor[pt[4] : shape[0] - pt[5], pt[2] : shape[1] - pt[3], pt[0] : shape[2] - pt[1]]
        return tensor

    def _pad_to_cover_all(self):

        # so the patches cover the entirity of the cube

        pad = []
        for idx, i in enumerate(self.cube.shape):
            if i < self.kernel_size[idx]:
                raise ValueError("CUBE DIM {} is smaller than corresponding size wanted for patch".format(idx))
            else:
                resto = self.cube.shape[idx] % self.kernel_size[idx]
                if resto == 0:
                    pad.insert(0, 0)
                    pad.insert(0, 0)
                elif resto % 2 == 0:
                    pad.insert(0, int(resto / 2))
                    pad.insert(0, int(resto / 2))
                else:
                    maior = int((resto - 1) / 2)
                    menor = int(resto - maior)
                    pad.insert(0, maior)
                    pad.insert(0, menor)

        if set(pad) == {0}:
            return

        self.pad_tuple = tuple(pad)  # store as attribute to then unpad
        self.cube = F.pad(self.cube, self.pad_tuple, "constant", 0)


if __name__ == "__main__":

    f = FullCubeSegmentationVisualizer(
        model_path="pretrained_weights/FROM_SCRATCH_cellari_heart_sup_2D_UNET_2D/only_supervised/run_1/weights_sup.pt",
        dataset_dir="pytorch/datasets/heart_mri/datasets/x_cubes_full",
        dataset_labels_dir="pytorch/datasets/heart_mri/datasets/y_cubes_full",
        dataset_name="heart_cellari",
    )
    f.get_segmentation_examples()

    # cube = np.load("pytorch/datasets/heart_mri/datasets/x_cubes_full/HeartData1_img_phase_2.npy")
    # p = Patcher(cube, two_dim=True)
    # for idx, patch in p:
    #    print(idx, patch.shape)

    # from utils import pad_if_necessary_one_array

    # a = torch.randn(1, 1, 64, 64, 12)
    # b, pt = pad_if_necessary_one_array(a, return_pad_tuple=True)
    # unp = FullCubeSegmentationConstructor._unpad_3d_array

    # c = unp(b, pt)
    # print((c == a).all())
