import os
from random import sample
import numpy as np
import SimpleITK as sitk
import torch

# from skimage.util.shape import view_as_windows
import torch.nn.functional as F
import nibabel
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from finetune import Trainer
from utils import *

# 3D Plotting
from skimage.measure import mesh_surface_area
from skimage.measure import marching_cubes_lewiner as marching_cubes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# from scipy.spatial import Delaunay
from scipy.ndimage.morphology import distance_transform_edt as dtrans


class FullCubeSegmentationConstructor:
    def __init__(self, model_path: str, dataset_dir: str, dataset_name: str):

        self.dataset_dir = dataset_dir  # original cubes, not extracted ones for training
        self.task_dir = "/".join(
            i for i in model_path.split("/")[1:-2]
        )  # eg model_dir: #pretrained_weights/FROM_pretrained_weights/PRETRAIN_MG_FRAMEWORK_task01_ss_VNET_MG/run_1__task01_sup_VNET_MG/only_supervised/run_1/weights_sup.pt
        self.config = get_config_object_of_task_dir(self.task_dir)
        self.dataset_name = dataset_name
        # self.dataset = get_dataset_object_of_task_dir(self.task_dir)

        # for key. value in dataset_map.items():
        #    if value == "/".join(self.dataset.x_data_dir.split("/")[:-2]:
        #        self.dataset_name = key

        assert self.config is not None
        # assert self.dataset is not None

        self.save_dir = os.path.join("viz_samples/", self.task_dir)
        make_dir(self.save_dir)

        self.trainer = Trainer(config=self.config, dataset=None)  # instanciating trainer to load and access model
        self.trainer.load_model(from_path=True, path=model_path, phase="sup")
        self.model = self.trainer.model

    def get_segmentation_examples(self, nr_cubes=3, save_slices=True):

        segmentations = []
        self.cubes_to_use = sample(os.listdir(self.dataset_dir), k=nr_cubes)
        self.cubes_to_use_path = [os.path.join(self.dataset_dir, i) for i in self.cubes_to_use]
        for cube_path in self.cubes_to_use_path:
            np_array = self._load_cube_to_np_array(cube_path)  # (x,y,z)
            patcher = Patcher(np_array)
            with torch.no_grad():
                self.model.eval()
                for idx, patch in patcher:
                    patch = torch.unsqueeze(patch, 0)  # (1,C,H,W) -> (1,1,C,H,W)
                    if self.config.model.lower() in ("vnet_mg", "unet_3d", "unet_acs"):
                        patch, _, pad_tuple = pad_if_necessary_one_array(patch, return_pad_tuple=True)
                    pred = self.model(patch)
                    # need to then unpad to reconstruct
                    pred = self._unpad_3d_array(pred, pad_tuple)
                    pred = torch.squeeze(dim=0)  # (1, 1, C,H,W) -> (1,C,H,W)
                    pred_mask = self._make_pred_mask_from_pred(pred)
                    patcher.predicitons_to_reconstruct_from[
                        :, idx
                    ] = pred_mask  # update array in patcher that will construct full cube predicted mask
            segmentations.append(patcher.get_pred_mask_full_cube())

        for idx, seg in enumerate(segmentations):
            # torch.save(seg, os.path.join(self.save_dir, self.cubes_to_use_path[idx]))
            save_dir = os.path.join(self.save_dir, self.dataset_name, "/")
            make_dir(save_dir)
            nifty_img = nibabel.Nifti1Image(np.array(seg).astype(np.float32), np.eye(4))
            nibabel.save(nifty_img, os.path.join(save_dir, self.cubes_to_use[idx]))
            self.save_3d_plot(np.array(seg), os.path.join(save_dir, "{}_plt3d.png".format(self.cubes_to_use[idx])))
            if save_slices is True:
                for z_idx in range(seg.shape[-1]):
                    fig = plt.figure(figsize=(10, 5))
                    plt.imshow(seg[:, :, z_idx], cmap=cm.Greys_r)
                    fig.savefig(
                        os.path.join(save_dir, self.cubes_to_use[idx], "/slices/" "slice_{}.jpg".format(z_idx)),
                        bbox_inches="tight",
                        dpi=150,
                    )

    @staticmethod
    def _unpad_3d_array(tensor, pad_tuple):
        # if pad(2,3) then [2:x-3]
        assert len(pad_tuple) == 2 * len(tensor.size())
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

    def __init__(self, cube: np.ndarray or torch.Tensor):

        self.cube = cube
        self._build_patches()

    def __iter__(self):

        for idx in range(self.patches.size()[1]):
            yield idx, self.patches[:, idx]  # (1, C, H, W)

    def _build_patches(self):

        cube_dimensions = self.cube.shape
        print("CUBE DIMENSIONS: {}".format(cube_dimensions))

        if len(cube_dimensions) == 3:

            self.kernel_size = [64, 64, 32]  # kernel size
            stride = [64, 64, 32]  # stride
            for idx in range(len(self.kernel_size)):
                if self.kernel_size[idx] > cube_dimensions[idx]:
                    self.kernel_size[idx] = cube_dimensions[idx]
                    stride[idx] = self.kernel_size[idx]

            self.cube = torch.Tensor(self.cube)
            self._pad_to_cover_all()
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

            # prepare tensor that is to be updsted with model predictions
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
        assert patches_orig.size() == self.cube.size()
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
    pass
    # cube = np.load("pytorch/datasets/heart_mri/datasets/x_cubes_full/HeartData1_img_phase_2.npy")
    # p = Patcher(cube)
    # for idx, patch in p:
    #    print(idx, patch.shape)

    # from utils import pad_if_necessary_one_array

    # a = torch.randn(1, 1, 64, 64, 12)
    # b, pt = pad_if_necessary_one_array(a, return_pad_tuple=True)
    # unp = FullCubeSegmentationConstructor._unpad_3d_array

    # c = unp(b, pt)
    # print((c == a).all())
