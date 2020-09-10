from pylidc.utils import consensus
import pylidc as pl
import numpy as np
import random
from skimage.transform import resize
import os


def make_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_padding_to_make_cube_80_80_80_as_in_lidc(shape):

    wanted = (80, 80, 80)
    slack = []
    for i, j in zip(wanted, shape):
        assert i > j
        slack.insert(0, (i - j))

    pad = []
    for i in slack:
        if i % 2 == 0:
            pad.insert(0, (int(i / 2), int(i / 2)))
        else:
            maior = int((i - 1) / 2)
            menor = int(i - maior)
            pad.insert(0, (maior, menor))

    return pad


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
            assert ((crop_window_target == 0) | (crop_window_target == 1)).all(), "Target array is not binary"

        if config.crop_rows != config.input_rows or config.crop_cols != config.input_cols:
            crop_window = resize(
                crop_window, (config.input_rows, config.input_cols, config.input_deps + config.len_depth), preserve_range=True,
            )
            if target_array:
                crop_window_target = resize(
                    crop_window_target, (config.input_rows, config.input_cols, config.input_deps + config.len_depth), preserve_range=True,
                )

        # skip "full" tissues
        if np.count_nonzero(crop_window) > (0.99 * crop_window.size):
            print("SKIPPING FULL TISSUE")
            continue
        # skip "air" cubes
        elif np.count_nonzero(crop_window) < (0.01 * crop_window.wize):
            print("SKIPPING AIR CUBE")
            continue
        else:
            cnt += 1
            num_pair += 1

        slice_set[num_pair] = crop_window[:, :, : config.input_deps]
        if target_array is not None:
            slice_set_target[num_pair] = crop_window_target[:, :, : config.input_deps]

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

    X_DIR = "/work2/s182312/lidc_mini_cubes_for_acs_replication/x"
    Y_DIR = "/work2/s182312/lidc_mini_cubes_for_acs_replication/y"

    make_dir(X_DIR)
    make_dir(Y_DIR)

    # print("ADJUSTING MIN")

    for i, scan in enumerate(scans):
        nodules = scan.cluster_annotations()

        hu_max, hu_min = 1000, -1000
        vol = scan.to_volume()
        mask_vol = np.zeros((vol.shape))
        # go over each annotaded nodule
        while np.max(vol) < hu_max:
            hu_max -= 10
        # print("ADJUSTING MAX")
        while np.min(vol) > hu_min:
            hu_min += 10

        if hu_max != 0 and hu_min != 0:
            vol[vol < hu_min] = hu_min
            vol[vol > hu_max] = hu_max
            vol = 1.0 * (vol - hu_min) / (hu_max - hu_min)

        for j, annotations_of_a_nodule in enumerate(nodules):
            first_annotation = annotations_of_a_nodule[0]  # select the first anotation
            first_annotation_mask = first_annotation.boolean_mask()
            padding = get_padding_to_make_cube_80_80_80_as_in_lidc(first_annotation_mask.shape)
            first_annotation_mask = annotations_of_a_nodule.boolean_mask(pad=padding)
            first_annotation_bbox = annotations_of_a_nodule[0].bbox(pad=padding)
            x = vol[first_annotation_bbox]
            y = first_annotation_mask
            assert np.count_nonzero(y) > 0
            np.save(os.path.join(X_DIR, "{}_{}.npy".format(scan.patient_id[-4:], j), x))
            np.save(os.path.join(Y_DIR, "{}_{}.npy".format(scan.patient_id[-4:], j), y))

    # slice_set_tr_array = np.array(slice_set_tr)
    # slice_set_tr_target_array = np.array(slice_set_tr_target)
    # slice_set_val_array = np.array(slice_set_val)
    # slice_set_val_target_array = np.array(slice_set_val_target)
    # slice_set_ts_array = np.array(slice_set_ts)
    # slice_set_ts_target_array = np.array(slice_set_ts_target)

    """ print(slice_set_tr_array.shape)
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
    ) """

