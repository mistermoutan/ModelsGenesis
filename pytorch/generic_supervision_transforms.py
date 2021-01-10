import torch
import torchio as tio
import random
import numpy as np


class TransformsForSupervision:
    def __init__(self, use_tio_flip=True):

        self.flip = tio.RandomFlip(p=0.5) if use_tio_flip is True else Flip3D()
        self.affine = tio.RandomAffine(p=0.5, scales=0.1, degrees=5, translation=0, image_interpolation="nearest")
        self.random_noise = tio.RandomNoise(p=0.5, std=(0, 0.1), include=["x"])  # don't apply noise to mask
        self.transform = tio.Compose([self.flip, self.affine, self.random_noise], include=["x", "y"])

    def __call__(self, x, binary_target):

        self._ensure_dimensions(x)
        x = np.squeeze(x, axis=0)  # comes as (B=1,C,H,W,D)
        binary_target = np.squeeze(binary_target, axis=0)
        d = {"x": x, "y": binary_target}
        res = self.transform(d)
        return np.expand_dims(res["x"], axis=(0)).astype(np.float32), np.expand_dims(res["y"], axis=(0)).astype(np.float32)

    @staticmethod
    def _ensure_dimensions(cube, dim=4):
        # assert 4D input
        assert len(cube.shape) == dim
        assert cube.shape[0] == 1  # 1 channel


class Flip3D:
    # unused
    possible_flip_axis = (0, 1, 2)

    def __call__(self, d: dict, p=0.5):
        x, y = d["x"], d["y"]
        axis_to_be_flipped = (p > torch.rand(3)).tolist()  # bool list
        for idx, i in enumerate(axis_to_be_flipped):
            if i is True:
                assert idx in self.__class__.possible_flip_axis
                x = np.flip(x, axis=idx)
                y = np.flip(y, axis=idx)

        return d