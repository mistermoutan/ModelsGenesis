import torch.nn as nn
from .base_converter import BaseConverter
from ..operators import ACSConv


class ACSConverter(BaseConverter):
    """
    Decorator class for converting 2d convolution modules
    to corresponding acs version in any networks.

    Args:
        model (torch.nn.module): model that needs to be converted
    Warnings:
        Functions in torch.nn.functional involved in data dimension are not supported
    Examples:
        >>> import ACSConverter
        >>> import torchvision
        >>> # m is a standard pytorch model
        >>> m = torchvision.models.resnet18(True)
        >>> m = ACSConverter(m)
        >>> # after converted, m is using ACSConv and capable of processing 3D volumes
        >>> x = torch.rand(batch_size, in_channels, D, H, W)
        >>> out = m(x)
    """

    converter_attributes = ["model"]
    target_conv = ACSConv

    def __init__(self, model, acs_kernel_split=None):
        """ Save the weights, convert the model to ACS counterpart, and then reload the weights """
        preserve_state_dict = model.state_dict()
        print("PROPORTIONS OF KERNEL SPLIT: {}".format(acs_kernel_split))
        model = self.convert_module(model, acs_kernel_split=acs_kernel_split)
        model.load_state_dict(preserve_state_dict, strict=False)  #
        self.model = model

    def convert_conv_kwargs(self, kwargs):
        kwargs["bias"] = True if kwargs["bias"] is not None else False
        kwargs["return_splits"] = True if kwargs.get("return_splits", False) is not False else False
        return kwargs