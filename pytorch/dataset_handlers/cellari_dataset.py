import os
import numpy as np


import matplotlib.pyplot as plt


def make_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


class CellariHeartDataset:
    def __init__(self, data_dir: str):

        self.data_dir = data_dir
        make_dir(os.path.join(data_dir, "x_cubes_full_test/"))
        make_dir(os.path.join(data_dir, "y_cubes_full_test/"))
        self.data_folders = os.listdir(data_dir)
        self.input_folders = [
            os.path.join(data_dir, i, "input_data", "input_data_raw")
            for i in self.data_folders
            if ("x_cubes" not in i and "y_cubes" not in i)
        ]
        self.target_folders = [
            os.path.join(data_dir, i, "input_data", "input_masks_annotated") for i in self.data_folders if i not in ("x", "y")
        ]
        print(self.input_folders, "\n", self.target_folders)

    def make_cubes(self):

        for input_folder, target_folder in zip(self.input_folders, self.target_folders):
            if "heartdata4-test-dataset,165" not in input_folder:
                continue
            xs = os.listdir(input_folder)
            ys = os.listdir(target_folder)
            for idx, i in enumerate(xs):
                splits = i.split("_")
                a = "_".join(i for i in splits[:-1])
                b = int(splits[-1][:-4])  # slice nr
                xs[idx] = (a, b, i)
            for idx, i in enumerate(ys):
                splits = i.split("_")
                a = "_".join(i for i in splits[:-1])
                b = int(splits[-1][:-4])
                ys[idx] = (a, b, i)

            ys.sort()
            xs.sort()
            ys = [i[-1] for i in ys]
            xs = [i[-1] for i in xs]

            assert len(xs) % 12 == 0 and len(ys) % 12 == 0 and len(xs) == len(ys)
            nr_cubes = int(len(xs) / 12)
            print("{} CUBES".format(nr_cubes))
            np_arrays_x = [[np.zeros((480, 480, 12))] for i in range(nr_cubes)]
            np_arrays_y = [[np.zeros((480, 480, 12))] for i in range(nr_cubes)]
            for idx, (x_, y_) in enumerate(zip(xs, ys)):
                assert x_ == y_, "files have same name rapaz \n {} {}".format(x_, y_)
                x = plt.imread(os.path.join(os.path.join(input_folder, x_)))
                y = plt.imread(os.path.join(os.path.join(target_folder, y_)))
                y = self._get_proper_slice_of_y(y, os.path.join(target_folder, y_))

                np_arrays_x[idx // 12][0][:, :, idx % 12] = x
                if len(np_arrays_x[idx // 12]) == 1:
                    np_arrays_x[idx // 12].append(x_)

                np_arrays_y[idx // 12][0][:, :, idx % 12] = y
                if len(np_arrays_y[idx // 12]) == 1:
                    np_arrays_y[idx // 12].append(y_)

            # save cubes as numpy arrays
            for array_name_tuple_x, array_name_tuple_y in zip(np_arrays_x, np_arrays_y):
                x_array = array_name_tuple_x[0]
                x_file_name = array_name_tuple_x[1]
                y_array = array_name_tuple_y[0]
                y_file_name = array_name_tuple_y[1]
                assert y_file_name == x_file_name
                split = x_file_name.split("_")
                name_to_save = "_".join(i for i in split[:-2])
                np.save(os.path.join(self.data_dir, "x_cubes_full_test/", "{}.npy".format(name_to_save)), x_array)
                np.save(os.path.join(self.data_dir, "y_cubes_full_test/", "{}.npy".format(name_to_save)), y_array)

    def _get_proper_slice_of_y(self, y_array, y_name=None):

        assert y_array.shape == (480, 480, 4)

        if 1 in y_array[:, :, 1]:
            assert False not in (y_array[:, :, 1] == y_array[:, :, 3]), "{}".format(y_name)
            assert 1 not in y_array[:, :, 0] and False not in (y_array[:, :, 0] == y_array[:, :, 2])
        else:
            for i in range(y_array.shape[-1] - 1):
                assert False not in (y_array[:, :, i] == y_array[:, :, i + 1])
            assert np.count_nonzero(y_array) == 0
        return y_array[:, :, 1]


if __name__ == "__main__":
    d = CellariHeartDataset("/home/moutan/Programming/thesis/ModelGenesis_Fork/ModelsGenesis/pytorch/datasets/heart_mri/datasets")
    d.make_cubes()
