import numpy as np

from ACSConv.experiments.mylib.voxel_transform import rotation, reflection, crop, random_center
from ACSConv.experiments.mylib.utils import _triple


class ACSPaperTransforms:
    def __init__(self, size, move=None, train=True):
        self.size = _triple(size)
        self.move = move
        self.train = train

    def __call__(self, voxel, seg):
        # workaround items come as (1,1,x, y, z) from get_train from dataset and this was originally designed for x,y,z input
        voxel = np.squeeze(voxel)
        seg = np.squeeze(seg)
        shape = voxel.shape
        # voxel = voxel / 255.0 - 1 already normalized
        if self.train:
            if self.move is not None:
                center = random_center(shape, self.move)
            else:
                center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
            seg_ret = crop(seg, center, self.size)

            angle = np.random.randint(4, size=3)
            voxel_ret = rotation(voxel_ret, angle=angle)
            seg_ret = rotation(seg_ret, angle=angle)

            axis = np.random.randint(4) - 1
            voxel_ret = reflection(voxel_ret, axis=axis)
            seg_ret = reflection(seg_ret, axis=axis)
        else:
            center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
            seg_ret = crop(seg, center, self.size)

        return np.expand_dims(voxel_ret, axis=(0, 1)).astype(np.float32), np.expand_dims(seg_ret, axis=(0, 1)).astype(np.float32)