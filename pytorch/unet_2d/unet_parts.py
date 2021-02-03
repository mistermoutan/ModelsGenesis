""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # print(type(x))
        # print(len(x))
        return self.double_conv(x)


class DoubleConvACS(nn.Module):
    """(convolution => [BN] => ReLU) * 2
    and return the kernel splits"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        setattr(self.conv2, "return_splits", True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, shape1, shape2, shape3 = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x, shape1, shape2, shape3


class DownACS(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels=None, return_splits=False):
        super().__init__()

        self.return_splits = return_splits
        self.maxpool = nn.MaxPool2d(2)
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        if self.return_splits is True:
            setattr(self.conv2, "return_splits", True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.return_splits is True:
            x = self.conv2(x)
            # print(type(x))
            x, shape1, shape2, shape3 = x[0], x[1], x[2], x[3]
        else:
            x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        if self.return_splits is False:
            return x
        else:
            return x, shape1, shape2, shape3


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class AxisAwareUpBlock(nn.Module):
    # Main question is to Where to use Conv3d we do not want to make number of parmeters a lot bigger

    def __init__(self, shapes: tuple, out_channels, channel_reduction_factor=2.5, mid_channels=None, bilinear=True):

        super(AxisAwareUpBlock, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        in_channels_a, in_channels_c, in_channels_s = shapes

        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.relu = nn.ReLU()

        mid_channels = int(in_channels_a // channel_reduction_factor)
        while mid_channels < out_channels:
            channel_reduction_factor -= 0.1
            mid_channels = int(in_channels_a // channel_reduction_factor)

        while any(in_channel < mid_channels for in_channel in [in_channels_a, in_channels_c, in_channels_s]):
            mid_channels -= 1

        # reduce filers before expensive conv3d as in going deeper with covolutions ala google inception
        print(
            "IN CHANNELS:{} {} {} ; MID CHANNELS:{}, OUT CHANNELS:{}".format(
                in_channels_a, in_channels_c, in_channels_s, mid_channels, out_channels
            )
        )
        self.conv1_a = nn.Conv3d(in_channels_a, mid_channels, kernel_size=1, padding=0)
        self.conv1_c = nn.Conv3d(in_channels_c, mid_channels, kernel_size=1, padding=0)
        self.conv1_s = nn.Conv3d(in_channels_s, mid_channels, kernel_size=1, padding=0)

        self.conv2_a = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.conv2_c = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.conv2_s = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)

        self.bn1_a = nn.BatchNorm3d(mid_channels)
        self.bn1_c = nn.BatchNorm3d(mid_channels)
        self.bn1_s = nn.BatchNorm3d(mid_channels)

        self.bn2_a = nn.BatchNorm3d(out_channels)
        self.bn2_c = nn.BatchNorm3d(out_channels)
        self.bn2_s = nn.BatchNorm3d(out_channels)

        self.bn3 = nn.BatchNorm3d(out_channels)

        self.global_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)
        setattr(self.global_conv, "return_splits", True)
        # self.global_conv = ACSConverter(self.global_conv)

        self.to(self.device)
        self.bn1_a.to(self.device)
        self.bn1_c.to(self.device)
        self.bn1_s.to(self.device)
        self.bn2_a.to(self.device)
        self.bn2_c.to(self.device)
        self.bn2_s.to(self.device)
        self.bn3.to(self.device)

    def forward(self, x_tuples: tuple):

        x_a, x_a_enc, x_c, x_c_enc, x_s, x_s_enc = x_tuples
        x_a = self.up(x_a)
        x_c = self.up(x_c)
        x_s = self.up(x_s)

        # skip connections
        x_a = self._match_padding_and_concat(x_a, x_a_enc)
        x_c = self._match_padding_and_concat(x_c, x_c_enc)
        x_s = self._match_padding_and_concat(x_s, x_s_enc)

        x_a = self.conv1_a(x_a)
        x_c = self.conv1_c(x_c)
        x_s = self.conv1_s(x_s)

        x_a = self.bn1_a(x_a)
        x_c = self.bn1_c(x_c)
        x_s = self.bn1_s(x_s)

        x_a = self.relu(x_a)
        x_c = self.relu(x_c)
        x_s = self.relu(x_s)

        x_a = self.conv2_a(x_a)
        x_c = self.conv2_c(x_c)
        x_s = self.conv2_s(x_s)

        x_a = self.bn2_a(x_a)
        x_c = self.bn2_c(x_c)
        x_s = self.bn2_s(x_s)

        x_a = self.relu(x_a)
        x_c = self.relu(x_c)
        x_s = self.relu(x_s)

        x = torch.cat([x_a, x_c, x_s], dim=1)
        x = self.global_conv(x)
        x, shape1, shape2, shape3 = x[0], x[1], x[2], x[3]
        x = self.bn3(x)
        x = self.relu(x)

        return x, shape1, shape2, shape3

    @staticmethod
    def _match_padding_and_concat(x1, x2) -> torch.Tensor:
        # padding on the tensors, not the ones that come from the encoder
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]
        pad_tuple = tuple([diffZ // 2, diffZ - diffZ // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffX - diffX // 2, 0, 0, 0, 0])
        x1 = F.pad(x1, pad_tuple)
        x1 = torch.cat([x2, x1], dim=1)
        return x1


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if len(x1.shape) == 5:  # 3D because ACS
            diffX = x2.size()[2] - x1.size()[2]
            diffY = x2.size()[3] - x1.size()[3]
            diffZ = x2.size()[4] - x1.size()[4]
            pad_tuple = tuple([diffZ // 2, diffZ - diffZ // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffX - diffX // 2, 0, 0, 0, 0])
            x1 = F.pad(x1, pad_tuple)

        elif len(x1.shape) == 4:  # input is BCHW
            diffX = x2.size()[2] - x1.size()[2]
            diffY = x2.size()[3] - x1.size()[3]
            pad_tuple = tuple([diffY // 2, diffY - diffY // 2, diffX // 2, diffX - diffX // 2, 0, 0, 0, 0])
            x1 = F.pad(x1, pad_tuple)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, sigmoid=False):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.apply_sigmoid = sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.conv(x) if self.apply_sigmoid is False else self.sigmoid(self.conv(x))
