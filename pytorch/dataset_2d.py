from numpy.random import shuffle
import numpy as np

from dataset import Dataset


class Dataset2D:
    def __init__(self, dataset3d, nr_slices_per_cube):

        self.dataset3d = dataset3d
        self.train_idxs, self.val_idxs = [], []
        self.nr_cubes_used = 0
        self.nr_slices_per_cube = nr_slices_per_cube  # to calculate len properly
        assert isinstance(nr_slices_per_cube, int)

    def get_train(self, batch_size: int) -> tuple():

        if not self.train_idxs:
            self.x_train_cube, self.y_train_cube = self.dataset3d.get_train(batch_size=1)  # (1,C,x,y,z)
            self.nr_cubes_used += 1
            print("nr_cubes_used: {}".format(self.nr_cubes_used))
            if self.x_train_cube is None:
                return (None, None)
            assert self.x_train_cube.shape[0] == 1 and self.x_train_cube.shape[1] == 1
            self.x_train_cube = self.x_train_cube[0]  # (1,C,x,y,z) -> (C,x,y,z)
            self.y_train_cube = self.y_train_cube[0]  # (1,C,x,y,z) -> (C,x,y,z)
            self.x_train_cube = np.transpose(self.x_train_cube, (-1, 0, 1, 2))  # (C,x,y,z) -> (z,C,x,y)
            self.y_train_cube = np.transpose(self.y_train_cube, (-1, 0, 1, 2))  # (C,x,y,z) -> (z,C,x,y)

            self.train_idxs = [i for i in range(self.x_train_cube.shape[0])]
            shuffle(self.train_idxs)

        x = self.x_train_cube[self.train_idxs[:batch_size]]
        if self.dataset3d.has_target:
            y = self.y_train_cube[self.train_idxs[:batch_size]]
        else:
            y = None

        del self.train_idxs[:batch_size]
        return (x, y)

    def get_val(self, batch_size: int) -> tuple():

        if not self.val_idxs:
            self.x_val_cube, self.y_val_cube = self.dataset3d.get_train(batch_size=1)  # (1,C,x,y,z)
            if self.x_val_cube is None:
                return (None, None)

            assert self.x_val_cube.shape[0] == 1 and self.x_val_cube.shape[1] == 1
            self.x_val_cube = self.x_val_cube[0]  # (1,C,x,y,z) -> (C,x,y,z)
            self.y_val_cube = self.y_val_cube[0]  # (1,C,x,y,z) -> (C,x,y,z)
            self.x_val_cube = np.transpose(self.x_val_cube, (-1, 0, 1, 2))  # (C,x,y,z) -> (z,C,x,y)
            self.y_val_cube = np.transpose(self.y_val_cube, (-1, 0, 1, 2))  # (C,x,y,z) -> (z,C,x,y)

            self.val_idxs = [i for i in range(self.x_val_cube.shape[0])]
            shuffle(self.train_idxs)

        x = self.x_val_cube[self.val_idxs[:batch_size]]
        if self.dataset3d.has_target:
            y = self.y_val_cube[self.val_idxs[:batch_size]]
        else:
            y = None

        del self.val_idxs[:batch_size]
        return (x, y)

    def get_len_train(self):
        nr_cubes_train = self.dataset3d.get_len_train()
        return nr_cubes_train * self.nr_slices_per_cube

    def get_len_val(self):
        nr_cubes_val = self.dataset3d.get_len_val()
        return nr_cubes_val * self.nr_slices_per_cube


if __name__ == "__main__":
    pass
    """     d3 = Dataset(
        data_dir="heart-mri-480-x-480,94 cellari/datasets/x_cubes_full/extracted_cubes_64_64_12_sup", train_val_test=(0.8, 0.2, 0)
    )
    d2 = Dataset2D(d3)

    cnt = 0
    while True:
        x, y = d2.get_train(batch_size=6)
        if x is None:
            break
        # print(x.shape)
        cnt += x.shape[0]
    print(cnt) """
