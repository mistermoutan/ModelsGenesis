import os
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict, Counter
import os
from collections import defaultdict
import torch
import nibabel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap

from dataset import Dataset
from utils import *
from finetune import Trainer
from full_cube_segmentation import FullCubeSegmentator

from unet_2d.unet_parts import AxisAwareUpBlock

feature_maps = {}


class FeatureExtractor:
    def __init__(self, config, dataset, test_all: bool = False):

        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dir = os.path.join("features/", self.config.task_dir)
        self.feature_plots_dir = os.path.join("feature_plots/", self.config.task_dir)
        self.kernel_splits_viz_dir = os.path.join("kernel_vizs/", self.config.task_dir)
        make_dir(self.feature_plots_dir)
        make_dir(self.feature_dir)
        make_dir(self.kernel_splits_viz_dir)
        model_path = os.path.join(self.config.model_path_save, "weights_sup.pt")
        self.task_dir = "/".join(i for i in model_path.split("/")[1:-1])

        self.dataset_name = None
        for key, value in dataset_map.items():
            if value == dataset.x_data_dir[:-3]:
                self.dataset_name = key
                break
        assert self.dataset_name is not None, "Could not find dataset name key in dataset_map dictionary"

    def plot_feature_maps_on_low_dimensional_space(self):
        # to: see if different axis features go on their own clusters
        feature_map_training = self._get_feature_map_training()
        self._plot_feature_maps_on_low_dimensional_space_phase(feature_map_training, "train")
        feature_map_testing = self._get_feature_map_testing()
        self._plot_feature_maps_on_low_dimensional_space_phase(feature_map_testing, "test")

    def plots_kernel_partitions_acs(self):
        pass
        # self.save_plots_kernel_partitions_acs()

    def _save_features_dataset(self, dataset):

        # to see: does each partition have different activations

        # how: plot rows with same feature maps from each arch

        # save all encoder and decoder features, also save sample name ?

        if dataset.x_test_filenames_original != []:
            previous_len = len(dataset.x_val_filenames_original)
            previous_val_filenames = deepcopy(dataset.x_val_filenames_original)
            dataset.x_val_filenames_original.extend(dataset.x_test_filenames_original)
            dataset.reset()
            assert len(dataset.x_val_filenames_original) == previous_len + len(dataset.x_test_filenames_original)
            assert previous_val_filenames != dataset.x_val_filenames_original

        # Hook to save feature maps https://discuss.pytorch.org/t/visualize-feature-map/29597
        global feature_maps

        # TESTING GET
        self._load_model()
        with torch.no_grad():
            self.model.eval()
            self.model.inc.register_forward_hook(self.get_activation(shapes=(22, 21, 21), layer_name="inc"))
            self.model.down1.register_forward_hook(self.get_activation(shapes=(43, 43, 42), layer_name="down1"))
            self.model.down2.register_forward_hook(self.get_activation(shapes=(86, 85, 85), layer_name="down2"))
            self.model.down3.register_forward_hook(self.get_activation(shapes=(171, 171, 170), layer_name="down3"))
            self.model.down4.register_forward_hook(self.get_activation(shapes=(171, 171, 170), layer_name="down4"))
            self.model.up1.register_forward_hook(self.get_activation(shapes=(86, 85, 85), layer_name="up1"))
            self.model.up2.register_forward_hook(self.get_activation(shapes=(43, 43, 42), layer_name="up2"))
            self.model.up3.register_forward_hook(self.get_activation(shapes=(22, 21, 21), layer_name="up3"))
            self.model.up4.register_forward_hook(self.get_activation(shapes=(22, 21, 21), layer_name="up4"))
            # self.model.up1.register_forward_hook(self.get_activation(shapes=(171,171,170), layer_name='outconv')) In out conv acs_split = (1,0,0) cause 1 channel

            while True:
                x, _ = dataset.get_val(batch_size=1, return_tensor=True)
                if x is None:
                    break
                x = x.float().to(self.device)
                if self.config.model.lower() in ("vnet_mg", "unet_3d", "unet_acs", "unet_acs_axis_aware_decoder", "unet_acs_with_cls"):
                    x, pad_tuple = pad_if_necessary_one_array(x, return_pad_tuple=True)
                    pred = self.model(x)
                    pred = FullCubeSegmentator._unpad_3d_array(pred, pad_tuple)
                elif "fcn_resnet18" in self.config.model.lower():
                    x = torch.cat((x, x, x), dim=1)
                    if 86 in x.shape:
                        continue
                    pred = self.model(x)
                else:
                    pred = self.model(x)

            dataset.reset()

        torch.save(feature_maps, os.path.join(self.feature_dir, "features_test.pt"))
        feature_maps.clear()

        # TRAINING SET
        with torch.no_grad():
            self.model.eval()
            while True:
                x, _ = dataset.get_train(batch_size=1, return_tensor=True)
                if x is None:
                    break
                x = x.float().to(self.device)

                if self.config.model.lower() in ("vnet_mg", "unet_3d", "unet_acs", "unet_acs_axis_aware_decoder", "unet_acs_with_cls"):
                    x, pad_tuple = pad_if_necessary_one_array(x, return_pad_tuple=True)
                    pred = self.model(x)
                    pred = FullCubeSegmentator._unpad_3d_array(pred, pad_tuple)
                elif "fcn_resnet18" in self.config.model.lower():
                    x = torch.cat((x, x, x), dim=1)
                    if 86 in x.shape:
                        continue
                    pred = self.model(x)
                else:
                    pred = self.model(x)
            dataset.reset()
        torch.save(feature_maps, os.path.join(self.feature_dir, "features_train.pt"))

    # also save acs slices of input? of 1 specific and only the features of that

    def _plot_feature_maps_on_low_dimensional_space_phase(self, features, phase):

        layers = ("inc", "down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4")
        for layer in layers:
            labels = []
            for key, value in features.items():
                if layer not in key:
                    continue
                features[key] = value.view(value.size(0), -1)  # flatten into (N, dims)
                if "_a" in key:
                    features_a = features[key]
                    labels.extend(["a" for _ in range(features_a.shape[0])])
                if "_c" in key:
                    features_c = features[key]
                    labels.extend(["a" for _ in range(features_c.shape[1])])
                if "_s" in key:
                    features_s = features[key]
                    labels.extend(["a" for _ in range(features_s.shape[2])])

            all_features = torch.cat([features_a, features_c, features_s], dim=0)
            all_features = all_features.cpu().numpy()
            self.draw_umap(all_features, labels, layer, phase=phase)

    def draw_umap(self, data, labels, layer_name: str, phase: str, n_neighbors=15, min_dist=0.1, metric="euclidean"):

        for n_components in (1, 2, 3):
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric=metric,
                random_state=42,
            )

            u = reducer.fit_transform(data)
            fig = plt.figure(figsize=(10, 5))

            if n_components == 1:
                ax = fig.add_subplot(111)
                ax.scatter(u[:, 0], range(len(u)), c=labels, cmap="Spectral", s=3)
            if n_components == 2:
                ax = fig.add_subplot(111)
                ax.scatter(u[:, 0], u[:, 1], c=labels, cmap="Spectral", s=3)
            if n_components == 3:
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=labels, s=100, cmap="Spectral")

            plt.gca().set_aspect("equal", "datalim")
            plt.colorbar(boundaries=np.arange(4) - 0.5).set_ticks(np.arange(3))
            fig.savefig(
                os.path.join(self.feature_plots_dir, "{}/".format(layer_name), "umap_{}d_projection_{}.jpg".format(n_components, phase)),
                bbox_inches="tight",
                dpi=150,
            )
            plt.close(fig=fig)

    def _get_feature_map_training(self):
        if os.path.isfile(os.path.join(self.feature_dir, "features_train.pt")):
            return torch.load(os.path.join(self.feature_dir, "features_train.pt"), map_location=self.device)
        else:
            self._save_features_dataset(self.dataset)
            return self._get_feature_map_training()

    def _get_feature_map_testing(self):
        if os.path.isfile(os.path.join(self.feature_dir, "features_test.pt")):
            return torch.load(os.path.join(self.feature_dir, "features_test.pt"), map_location=self.device)
        else:
            self._save_features_dataset(self.dataset)
            return self._get_feature_map_testing()

    def save_plots_kernel_partitions_acs(self):

        # also plot along and compare, kernels are 3x3 what we can do is UMAP as well on them
        self._load_model()

        kernels = {}

        if self.config.model.lower() == "unet_acs":
            kernels_inc = self.model.inc.double_conv[3].weight.detach()
            kernels_inc_a, kernels_inc_c, kernels_inc_s = kernels_inc[:22], kernels_inc[22 : 22 + 21], kernels_inc[22 + 21 :]

            kernels_down1 = self.model.down1.maxpool_conv[1].double_conv[3].weight.detach()
            kernels_down1_a, kernels_down1_c, kernels_down1_s = kernels_down1[:43], kernels_down1[43 : 43 + 43], kernels_down1[43 + 43 :]

            kernels_down2 = self.model.down2.maxpool_conv[1].double_conv[3].weight.detach()
            kernels_down2_a, kernels_down2_c, kernels_down2_s = kernels_down2[:86], kernels_down2[86 : 86 + 85], kernels_down2[86 + 85 :]

            kernels_down3 = self.model.down3.maxpool_conv[1].double_conv[3].weight.detach()
            kernels_down3_a, kernels_down3_c, kernels_down3_s = (
                kernels_down3[:171],
                kernels_down3[171 : 171 + 171],
                kernels_down3[171 + 171 :],
            )

            kernels_down4 = self.model.down4.maxpool_conv[1].double_conv[3].weight.detach()
            kernels_down4_a, kernels_down4_c, kernels_down4_s = (
                kernels_down4[:171],
                kernels_down4[171 : 171 + 171],
                kernels_down4[171 + 171 :],
            )

            kernels_up1 = self.model.up1.conv.double_conv[3].weight.detach()
            kernels_up1_a, kernels_up1_c, kernels_up1_s = kernels_up1[:86], kernels_up1[86 : 86 + 85], kernels_up1[86 + 85 :]

            kernels_up2 = self.model.up2.conv.double_conv[3].weight.detach()
            kernels_up2_a, kernels_up2_c, kernels_up2_s = kernels_up2[:43], kernels_up2[43 : 43 + 43], kernels_up2[43 + 43 :]

            kernels_up3 = self.model.up3.conv.double_conv[3].weight.detach()
            kernels_up3_a, kernels_up3_c, kernels_up3_s = kernels_up3[:22], kernels_up3[22 : 22 + 21], kernels_up3[22 + 21 :]

            kernels_up4 = self.model.up4.conv.double_conv[3].weight.detach()
            kernels_up4_a, kernels_up4_c, kernels_up4_s = kernels_up4[:22], kernels_up4[22 : 22 + 21], kernels_up4[22 + 21 :]

        else:
            kernels_inc = self.model.inc.double_conv[3].weight.detach()
            kernels_inc_a, kernels_inc_c, kernels_inc_s = kernels_down4[:22], kernels_down4[22 : 22 + 21], kernels_down4[22 + 21 :]

            kernels_down1 = self.model.down1.conv2.weight.detach()
            kernels_down1_a, kernels_down1_c, kernels_down1_s = kernels_down1[:43], kernels_down1[43 : 43 + 43], kernels_down1[43 + 43 :]

            kernels_down2 = self.model.down2.conv2.weight.detach()
            kernels_down2_a, kernels_down2_c, kernels_down2_s = kernels_down2[:86], kernels_down2[86 : 86 + 85], kernels_down2[86 + 85 :]

            kernels_down3 = self.model.down3.conv2.weight.detach()
            kernels_down3_a, kernels_down3_c, kernels_down3_s = (
                kernels_down3[:171],
                kernels_down3[171 : 171 + 171],
                kernels_down3[171 + 171 :],
            )

            kernels_down4 = self.model.down4.conv2.weight.detach()
            kernels_down4_a, kernels_down4_c, kernels_down4_s = (
                kernels_down4[:171],
                kernels_down4[171 : 171 + 171],
                kernels_down4[171 + 171 :],
            )

            if isinstance(self.model.up1, AxisAwareUpBlock):
                kernels_up1 = self.model.up1.global_conv.weight.detach()
            else:
                kernels_up1 = self.model.up1.conv.double_conv[3].weight.detach()

            kernels_up1_a, kernels_up1_c, kernels_up1_s = kernels_up1[:86], kernels_up1[86 : 86 + 85], kernels_up1[86 + 85 :]

            if isinstance(self.model.up2, AxisAwareUpBlock):
                kernels_up2 = self.model.up2.global_conv.weight.detach()
            else:
                kernels_up2 = self.model.up2.conv.double_conv[3].weight.detach()

            kernels_up2_a, kernels_up2_c, kernels_up2_s = kernels_up2[:43], kernels_up2[43 : 43 + 43], kernels_up2[43 + 43 :]

            if isinstance(self.model.up3, AxisAwareUpBlock):
                kernels_up3 = self.model.up3.global_conv.weight.detach()
            else:
                kernels_up3 = self.model.up3.conv.double_conv[3].weight.detach()

            kernels_up3_a, kernels_up3_c, kernels_up3_s = kernels_up3[:22], kernels_up3[22 : 22 + 21], kernels_up3[22 + 21 :]

            if isinstance(self.model.up4, AxisAwareUpBlock):
                kernels_up4 = self.model.up4.global_conv.weight.detach()
            else:
                kernels_up4 = self.model.up4.conv.double_conv[3].weight.detach()

            kernels_up4_a, kernels_up4_c, kernels_up4_s = kernels_up4[:22], kernels_up4[22 : 22 + 21], kernels_up4[22 + 21 :]

        self._plot_kernel_partitions_acs([kernels_inc_a, kernels_inc_c, kernels_inc_s], "inc")
        self._plot_kernel_partitions_acs([kernels_down1_a, kernels_down1_c, kernels_down1_s], "down1")
        self._plot_kernel_partitions_acs([kernels_down2_a, kernels_down2_c, kernels_down2_s], "down2")
        self._plot_kernel_partitions_acs([kernels_down3_a, kernels_down3_c, kernels_down3_s], "down3")
        self._plot_kernel_partitions_acs([kernels_down4_a, kernels_down4_c, kernels_down4_s], "down4")
        self._plot_kernel_partitions_acs([kernels_up1_a, kernels_up1_c, kernels_up1_s], "up1")
        self._plot_kernel_partitions_acs([kernels_up2_a, kernels_up2_c, kernels_up2_s], "up2")
        self._plot_kernel_partitions_acs([kernels_up3_a, kernels_up3_c, kernels_up3_s], "up3")
        self._plot_kernel_partitions_acs([kernels_up4_a, kernels_up4_c, kernels_up4_s], "up4")

    @staticmethod
    def _plot_kernel_partitions_acs(kernels: list, layer: str):
        # Visualize conv filter
        # feats = [kernels_a, kernels_c, kernels_s]
        for idx, filters in enumerate(kernels):
            if idx == 0:
                axis = "a"
            if idx == 1:
                axis = "c"
            if idx == 2:
                axis = "s"

            fig, axarr = plt.subplots(filters.size(0))
            for idx in range(filters.size(0)):
                axarr[idx].imshow(kernels[idx].squeeze())

    def _load_model(self):

        if not hasattr(self, "model"):
            trainer = Trainer(config=self.config, dataset=None)  # instanciating trainer to load and access model
            weight_files = os.listdir(self.config.model_path_save)
            if "weights_sup.pt" in weight_files:
                trainer.load_model(
                    from_path=True,
                    path=os.path.join(self.config.model_path_save, "weights_sup.pt"),
                    phase="sup",
                    ensure_sup_is_completed=True,
                    data_paralell=False,
                )
            elif "weights_ss.pt" in weight_files:
                trainer.load_model(
                    from_path=True, path=os.path.join(self.config.model_path_save, "weights_ss.pt"), phase="ss", data_paralell=False
                )
            else:
                raise ValueError

            self.model = trainer.model

    @staticmethod
    def get_activation(shapes: tuple, layer_name: str):
        # BUILD DICT, shape will be (N, Activatios/Feature Maps (1...N), spatial dims)
        def hook(model, input, output):
            if isinstance(output, tuple):
                o = output.detach()[0]
                assert isinstance(o, torch.Tensor)
            elif isinstance(output, torch.Tensor):
                o = output.detach()
            else:
                raise ValueError

            shape_a, shape_c, _ = shapes

            # first dim is batch
            if feature_maps.get("{}_a".format(layer_name), None) is None:
                feature_maps["{}_a".format(layer_name)] = o[:, :shape_a]
            else:
                feature_maps["{}_a".format(layer_name)] = torch.cat([feature_maps["{}_a".format(layer_name)], o[:, :shape_a]], dim=0)

            if feature_maps.get("{}_c".format(layer_name), None) is None:
                feature_maps["{}_c".format(layer_name)] = o[:, shape_a : shape_a + shape_c]
            else:
                feature_maps["{}_c".format(layer_name)] = torch.cat(
                    [feature_maps["{}_c".format(layer_name)], o[:, shape_a : shape_a + shape_c]], dim=0
                )

            if feature_maps.get("{}_s".format(layer_name), None) is None:
                feature_maps["{}_s".format(layer_name)] = o[:, shape_a + shape_c :]
            else:
                feature_maps["{}_s".format(layer_name)] = torch.cat(
                    [feature_maps["{}_s".format(layer_name)], o[:, shape_a + shape_c :]], dim=0
                )

        return hook

    """ @staticmethod
    def get_activation_down4(account_for_splits=False):
            # BUILD DICT
            def hook(model, input, output):

                if account_for_splits is True:
                    o = output.detach()[0]
                else:
                    o = output.detach()
                
                if feature_maps.get("down4_a", None) is None:
                    feature_maps["down4_a"] = o[:, :171]
                else:
                    feature_maps["down4_a"] = torch.cat([feature_maps["down4_a"], o[:, :171]])

                if feature_maps.get("down4_c", None) is None:
                    feature_maps["down4_c"] = o[:, 171:342]
                else:
                    feature_maps["down4_c"] = torch.cat([feature_maps["down4_c"], o[:, 171:342]])

                if feature_maps.get("down4_s", None) is None:
                    feature_maps["down4_s"] = o[:, 342:]
                else:
                    feature_maps["down4_s"] = torch.cat([feature_maps["down4_s"], o[:, 342:]])

            return hook
        
    @staticmethod
    def get_activation_down3(account_for_splits=False):
            # BUILD DICT
            def hook(model, input, output):
                if feature_maps.get("down3_a", None) is None:
                    feature_maps["down3_a"] = o[:, :171]
                else:
                    feature_maps["down3_a"] = torch.cat([feature_maps["down3_a"], o[:, :171]])

                if feature_maps.get("down3_c", None) is None:
                    feature_maps["down3_c"] = o[:, 171:342]
                else:
                    feature_maps["down3_c"] = torch.cat([feature_maps["down3_c"], o[:, 171:342]])

                if feature_maps.get("down3_s", None) is None:
                    feature_maps["down3_s"] = o[:, 342:]
                else:
                    feature_maps["down3_s"] = torch.cat([feature_maps["down3_s"], o[:, 342:]])

            return hook
    
    @staticmethod
    def get_activation_down2(account_for_splits=False):
        # BUILD DICT
        def hook(model, input, output):
            if feature_maps.get("down2_a", None) is None:
                feature_maps["down2_a"] = o[:, :86]
            else:
                feature_maps["down2_a"] = torch.cat([feature_maps["down2_a"], o[:, :86]])

            if feature_maps.get("down2_c", None) is None:
                feature_maps["down2_c"] = o[:, 86:86+85]
            else:
                feature_maps["down2_c"] = torch.cat([feature_maps["down2_c"], o[:, 86:86+85]])

            if feature_maps.get("down2_s", None) is None:
                feature_maps["down2_s"] = o[:, 86+85:]
            else:
                feature_maps["down2_s"] = torch.cat([feature_maps["down2_s"], o[:, 86+85:]])

        return hook

    @staticmethod
    def get_activation_down1(account_for_splits=False):
        # BUILD DICT
        def hook(model, input, output):

            if account_for_splits is True:
                o = output.detach()[0]
            else:
                o = output.detach()

            if feature_maps.get("down1_a", None) is None:
                feature_maps["down1_a"] = o[:, :43]
            else:
                feature_maps["down1_a"] = torch.cat([feature_maps["down1_a"], o[:, :43]])

            if feature_maps.get("down1_c", None) is None:
                feature_maps["down1_c"] = o[:, 43:86]
            else:
                feature_maps["down1_c"] = torch.cat([feature_maps["down1_c"], o[:, 43:86]])

            if feature_maps.get("down1_s", None) is None:
                feature_maps["down1_s"] = o[:, 86:]
            else:
                feature_maps["down1_s"] = torch.cat([feature_maps["down1_s"], o[:, 86:]])

        return hook

    @staticmethod
    def get_activation_inc(account_for_splits=False):
        # BUILD DICT
        def hook(model, input, output):
            if feature_maps.get("inc_a", None) is None:
                feature_maps["inc_a"] = o[:, :22]
            else:
                feature_maps["inc_a"] = torch.cat([feature_maps["inc_a"], o[:, :22]])

            if feature_maps.get("inc_c", None) is None:
                feature_maps["inc_c"] = o[:, 22:43]
            else:
                feature_maps["inc_c"] = torch.cat([feature_maps["inc_c"], o[:, 22:43]])

            if feature_maps.get("inc_s", None) is None:
                feature_maps["inc_s"] = o[:, 43:]
            else:
                feature_maps["inc_s"] = torch.cat([feature_maps["inc_s"], o[:, 43:]])

        return hook """