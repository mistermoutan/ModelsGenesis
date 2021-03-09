import numpy as np
import torch
import torch.nn.functional as F


class Patcher:

    # https://discuss.pytorch.org/t/patch-making-does-pytorch-have-anything-to-offer/33850/10

    def __init__(self, cube: np.ndarray or torch.Tensor, two_dim=False):

        self.cube = cube
        self.two_dim = two_dim
        self.original_cube_dimensions = self.cube.shape
        self._build_patches()

    def __iter__(self):

        for idx in range(self.patches.size()[1]):
            yield idx, self.patches[:, idx]  # (1, C, H, W)

    def _build_patches(self):

        print("CUBE DIMENSIONS: {}".format(self.original_cube_dimensions))

        if self.two_dim is False:
            self.kernel_size = [100, 100, 100]  # kernel size
            stride = self.kernel_size  # stride
        else:
            # patches for two dimensional model
            raise ValueError("Should not use patching in two dim as there should not be memory issues")
        dif = [0, 0, 0]
        for idx in range(len(self.kernel_size)):
            if self.kernel_size[idx] > self.original_cube_dimensions[idx]:
                dif[idx] += self.kernel_size[idx] - self.original_cube_dimensions[idx]
                self.kernel_size[idx] = self.original_cube_dimensions[idx]
                stride[idx] = self.kernel_size[idx]

        for idx, d in enumerate(dif):
            if d > 0:
                for k_idx, _ in enumerate(self.kernel_size):
                    if k_idx == idx:
                        continue
                    self.kernel_size[k_idx] += int(d / 2)
                    stride[k_idx] = self.kernel_size[k_idx]

        # recheck patch dimensions due to the dif that was done before
        for idx in range(len(self.kernel_size)):
            if self.kernel_size[idx] > self.original_cube_dimensions[idx]:
                self.kernel_size[idx] = self.original_cube_dimensions[idx]
                stride[idx] = self.kernel_size[idx]

        print("KERNEL: ", self.kernel_size)

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
        # get back to original cube
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
                base = self.kernel_size[idx]
                tmp = base
                multiplier = 2
                while tmp < self.cube.shape[idx]:
                    tmp = base * multiplier
                    multiplier += 1
                base = tmp
                resto = base - self.cube.shape[idx]
                # resto = self.cube.shape[idx] % self.kernel_size[idx]
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
    # p = Patcher(cube, two_dim=True)
    # for idx, patch in p:
    #    print(idx, patch.shape)
