from pylidc.utils import consensus
import pylidc as pl
import numpy as np
import random
from skimage.transform import resize


class setup_config:
    def __init__(
        self,
        input_rows=None,
        input_cols=None,
        input_deps=None,
        crop_rows=None,
        crop_cols=None,
        len_border=None,
        len_border_z=None,
        scale=None,
        DATA_DIR=None,
        len_depth=None,
        modality=None,
        target_dir=None,
    ):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_deps = input_deps
        self.crop_rows = crop_rows
        self.crop_cols = crop_cols
        self.len_border = len_border
        self.len_border_z = len_border_z
        self.scale = scale
        self.DATA_DIR = DATA_DIR
        self.len_depth = len_depth
        if modality == "ct":
            self.hu_max, self.hu_min = 1000, -1000
        if modality == "mri":
            self.hu_max, self.hu_min = 4000, 0
        self.target_dir = target_dir


def infinite_generator_from_one_volume(config, img_array, target_array=None):

    size_x, size_y, size_z = img_array.shape

    if size_z - config.input_deps - config.len_depth - 1 - config.len_border_z < config.len_border_z:
        return None

    # min-max normalization
    img_array[img_array < config.hu_min] = config.hu_min
    img_array[img_array > config.hu_max] = config.hu_max
    img_array = 1.0 * (img_array - config.hu_min) / (config.hu_max - config.hu_min)
    slice_set = np.zeros((config.scale, config.input_rows, config.input_cols, config.input_deps), dtype=float)
    slice_set_target = np.zeros((config.scale, config.input_rows, config.input_cols, config.input_deps), dtype=float)

    num_pair = 0
    cnt = 0

    while True:
        cnt += 1
        if cnt > 50 * config.scale and num_pair == 0:
            return None
        elif cnt > 50 * config.scale and num_pair > 0:
            return np.array(slice_set[:num_pair])

        if hasattr(config, "starting_x"):

            x_interval = config.ending_x - config.starting_x
            y_interval = config.ending_y - config.starting_y
            z_interval = config.ending_z - config.starting_z
            x_slack = config.crop_rows - x_interval
            y_slack = config.crop_cols - y_interval
            z_slack = config.input_deps - z_interval

            if x_slack >= 0:  # then make sure you get everything as it is possible
                start_x = random.randint(config.starting_x - x_slack // 2, config.starting_x)
                while 1 not in target_array[start_x : start_x + config.crop_rows, :, :]:
                    print("this should not happen X")
                    start_x = random.randint(config.starting_x - x_slack // 2, config.starting_x)
                slack_border_x = size_x - start_x
                if slack_border_x < config.input_rows:
                    start_x -= config.input_rows - slack_border_x
            else:
                start_x = random.randint(config.starting_x, config.starting_x - x_slack // 2)  # slack is negative
                while 1 not in target_array[start_x : start_x + config.crop_rows, :, :]:
                    print("FINDING NEW X STARTING POINT")
                    start_x = random.randint(config.starting_x, config.starting_x - x_slack // 2)

            if y_slack >= 0:  # then make sure you get everything as it is possible
                start_y = random.randint(config.starting_y - y_slack // 2, config.starting_y)
                while 1 not in target_array[:, start_y : start_y + config.crop_cols, :]:
                    print("this should not happen Y")
                    start_y = random.randint(config.starting_y - y_slack // 2, config.starting_y)
                slack_border_y = size_y - start_y
                if slack_border_y < config.input_cols:
                    start_y -= config.input_cols - slack_border_y
            else:
                start_y = random.randint(config.starting_y, config.starting_y - y_slack // 2)
                while 1 not in target_array[:, start_y : start_y + config.crop_cols, :]:
                    print("FINDING NEW Y STARTING POINT")
                    start_y = random.randint(config.starting_y, config.starting_y - y_slack // 2)

            if z_slack >= 0:  # then make sure you get everything as it is possible
                start_z = random.randint(config.starting_z - z_slack // 2, config.starting_z)
                while 1 not in target_array[:, :, start_z : start_z + config.input_deps]:
                    print("this should not happen z")
                    start_z = random.randint(config.starting_z - z_slack // 2, config.starting_z)
                slack_border_z = size_z - start_z
                if slack_border_z < config.input_deps:
                    start_z -= config.input_deps - slack_border_z
            else:
                start_z = random.randint(config.starting_z, config.starting_z - z_slack // 2)
                while 1 not in target_array[:, :, start_z : start_z + config.input_deps]:
                    print("FINDING NEW Z STARTING POINT")
                    start_z = random.randint(config.starting_z, config.starting_z - z_slack // 2)

        else:

            start_x = random.randint(0 + config.len_border, size_x - config.crop_rows - 1 - config.len_border)
            start_y = random.randint(0 + config.len_border, size_y - config.crop_cols - 1 - config.len_border)
            start_z = random.randint(0 + config.len_border_z, size_z - config.input_deps - config.len_depth - 1 - config.len_border_z)

        # get the cube
        crop_window = img_array[
            start_x : start_x + config.crop_rows,
            start_y : start_y + config.crop_cols,
            start_z : start_z + config.input_deps + config.len_depth,
        ]

        if target_array is not None:
            assert type(target_array) == np.ndarray
            crop_window_target = target_array[
                start_x : start_x + config.crop_rows,
                start_y : start_y + config.crop_cols,
                start_z : start_z + config.input_deps + config.len_depth,
            ]

        if config.crop_rows != config.input_rows or config.crop_cols != config.input_cols:
            crop_window = resize(
                crop_window, (config.input_rows, config.input_cols, config.input_deps + config.len_depth), preserve_range=True,
            )
            if target_array:
                crop_window_target = resize(
                    crop_window_target,
                    (config.input_rows, config.input_cols, config.input_deps + config.len_depth),
                    preserve_range=True,
                )

        slice_set[num_pair] = crop_window[:, :, : config.input_deps]
        if target_array is not None:
            slice_set_target[num_pair] = crop_window_target[:, :, : config.input_deps]

        num_pair += 1
        if num_pair == config.scale:
            break

    if target_array is None:
        return np.array(slice_set)
    else:
        return np.array(slice_set), np.array(slice_set_target)


if __name__ == "__main__":

    scans = pl.query(pl.Scan).all()
    config = setup_config(
        input_rows=64,
        input_cols=64,
        input_deps=32,
        crop_rows=64,
        crop_cols=64,
        scale=1,
        len_border=100,
        len_border_z=20,
        len_depth=3,
        modality="ct",
    )
    padding = [(0, 0), (0, 0), (0, 0)]

    slice_set_tr = []  # np.zeros((510, config.input_rows, config.input_cols, config.input_deps), dtype=float)
    slice_set_tr_target = []  # np.zeros((510, config.input_rows, config.input_cols, config.input_deps), dtype=float)
    slice_set_val = []  # np.zeros((100, config.input_rows, config.input_cols, config.input_deps), dtype=float)
    slice_set_val_target = []  # np.zeros((100, config.input_rows, config.input_cols, config.input_deps), dtype=float)
    slice_set_ts = []  # np.zeros((408, config.input_rows, config.input_cols, config.input_deps), dtype=float)
    slice_set_ts_target = []  # np.zeros((408, config.input_rows, config.input_cols, config.input_deps), dtype=float)

    for i, scan in enumerate(scans):
        nodules = scan.cluster_annotations()
        # for p in nodules:
        #    if len(p) > 4:
        #        print("HELLO")
        vol = scan.to_volume()
        mask_vol = np.zeros((vol.shape))
        # go over each annotaded nodule
        for j, annotations_of_a_nodule in enumerate(nodules):
            cons_mask, cons_bbox = consensus(annotations_of_a_nodule, ret_masks=False, pad=padding)
            mask_vol[cons_bbox] = cons_mask
            config.starting_x, config.ending_x = cons_bbox[0].start, cons_bbox[0].stop
            config.starting_y, config.ending_y = cons_bbox[1].start, cons_bbox[1].stop
            config.starting_z, config.ending_z = cons_bbox[2].start, cons_bbox[2].stop
            res = infinite_generator_from_one_volume(config, vol, mask_vol)
            if res is None:
                print("NONE")
                continue
            else:
                x, y = res
            if i < 510:
                slice_set_tr.extend(x)
                slice_set_tr_target.extend(y)
            elif 510 <= i < 610:
                slice_set_val.extend(x)
                slice_set_val_target.extend(y)
            else:
                slice_set_ts.extend(x)
                slice_set_ts_target.extend(x)

        if i % 50 == 0:
            print(i)

    slice_set_tr_array = np.array(slice_set_tr)
    slice_set_tr_target_array = np.array(slice_set_tr_target)
    slice_set_val_array = np.array(slice_set_val)
    slice_set_val_target_array = np.array(slice_set_val_target)
    slice_set_ts_array = np.array(slice_set_ts)
    slice_set_ts_target_array = np.array(slice_set_ts_target)

    print(slice_set_tr_array.shape)
    print(slice_set_tr_target_array.shape)
    print(slice_set_val_array.shape)
    print(slice_set_val_target_array.shape)
    print(slice_set_ts_array.shape)
    print(slice_set_ts_target_array.shape)

    np.save(
        "tr_cubes" + "_" + str(config.input_rows) + "x" + str(config.input_cols) + "x" + str(config.input_deps) + ".npy",
        slice_set_tr_array,
    )
    np.save(
        "tr_cubes" + "_" + str(config.input_rows) + "x" + str(config.input_cols) + "x" + str(config.input_deps) + "_target.npy",
        slice_set_tr_target_array,
    )
    np.save(
        "val_cubes" + "_" + str(config.input_rows) + "x" + str(config.input_cols) + "x" + str(config.input_deps) + ".npy",
        slice_set_val_array,
    )
    np.save(
        "val_cubes" + "_" + str(config.input_rows) + "x" + str(config.input_cols) + "x" + str(config.input_deps) + "_target.npy",
        slice_set_val_target_array,
    )
    np.save(
        "ts_cubes" + "_" + str(config.input_rows) + "x" + str(config.input_cols) + "x" + str(config.input_deps) + ".npy",
        slice_set_ts_array,
    )
    np.save(
        "ts_cubes" + "_" + str(config.input_rows) + "x" + str(config.input_cols) + "x" + str(config.input_deps) + "_target.npy",
        slice_set_ts_target_array,
    )

