""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from torch import nn

from .unet_parts import *
from random import choice

from ACSConv.acsconv.converters import ACSConverter


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, apply_sigmoid_to_output=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes) if apply_sigmoid_to_output is False else OutConv(64, n_classes, sigmoid=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x4.shape, x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UnetACSWithClassifier(nn.Module):
    "To be used with ACS conversion only"

    def __init__(self, n_channels, n_classes, bilinear=True, apply_sigmoid_to_output=False):
        super(UnetACSWithClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownACS(64, 128)
        self.down2 = DownACS(128, 256)
        self.down3 = DownACS(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownACS(512, 1024 // factor, return_splits=True)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes) if apply_sigmoid_to_output is False else OutConv(64, n_classes, sigmoid=True)

        self.fc1 = nn.Linear(170, 3, bias=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5, shape1, shape2, shape3 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        # do the classificationa as well
        # x_a, x_c, x_s = x5[:, :shape1], x5[:, shape1 : shape1 + shape2], x5[:, shape1 + shape2 :]
        x_cls, targets, nr_feat = self._get_data_for_classification(last_down_features=x5, shape1=shape1, shape2=shape2, shape3=shape3)
        # print("NUMBE FEAT:", nr_feat)
        # make spatial dimensions unitary and flatten them into (B,N)
        x_cls = F.adaptive_avg_pool3d(x_cls, output_size=1).view(x_cls.size(0), -1)
        x_cls = self.fc1(x_cls)
        return (logits, x_cls, targets)

    def _get_data_for_classification(self, last_down_features, shape1, shape2, shape3):

        x5 = last_down_features.detach().clone()  # (B,C,H,W,D)
        x5 = x5.to(self.device)
        batch_size = x5.shape[0]
        # equal split in relation to batch of how many examples from each view go to classifier
        # so from the elemensts of btach, how many featuers from each view will follow
        each_view = [batch_size // 3 for _ in range(3)]

        while sum(each_view) != batch_size:
            increment_choice = choice([0, 1, 2])
            each_view[increment_choice] += 1
        assert sum(each_view) == batch_size
        if 0 in each_view:
            print("EACH VIEW {}, because batch is of size {}".format(each_view, batch_size))

        # features belonging to each view
        x_a, x_c, x_s = x5[:, :shape1], x5[:, shape1 : shape1 + shape2], x5[:, shape1 + shape2 :]
        allowed_number_features = min(x_a.shape[1], x_c.shape[1], x_s.shape[1])  # due to kernel split we can have eg: 170,171,171

        res = []
        # take from each view's features the elements of batch that will follow to classification as each batch element has 3 views and only one will be used
        res.append(x_a[: each_view[0]])
        res.append(x_c[each_view[0] : each_view[0] + each_view[1]])
        res.append(x_s[each_view[0] + each_view[1] :])

        # make all have same features nr as in allowed_number_features
        for idx, i in enumerate(res):
            if i.shape[1] != allowed_number_features:
                i = torch.narrow(i, 1, 1, i.shape[1] - 1)
                res[idx] = i

        targets = torch.LongTensor([0 for i in range(each_view[0])] + [1 for i in range(each_view[1])] + [2 for i in range(each_view[2])])
        res_out = torch.cat(res, dim=0)  # concatenate on batch
        # print(res_out.shape)
        return res_out, targets, allowed_number_features


class UnetACSWithClassifierOnly(nn.Module):
    "To be used with ACS conversion only"

    def __init__(
        self,
        n_channels,
        n_classes,
        bilinear=True,
        apply_sigmoid_to_output=False,
        freeze_encoder=False,
        pool_features=False,
        encoder_depth=4,
    ):
        super(UnetACSWithClassifierOnly, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.encoder_depth = encoder_depth
        self.freeze_encoder = freeze_encoder
        self.pool_features = pool_features
        self.check_against = None

        print("ENCODER DEPTH FROM WHICH FEATURES WILL COME: {}".format(encoder_depth))
        if self.pool_features is True:
            print("GOING TO POOL FEATURES")

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownACS(64, 128, return_splits=True)
        self.down2 = DownACS(128, 256, return_splits=True)
        self.down3 = DownACS(256, 512, return_splits=True)
        factor = 2 if bilinear else 1
        self.down4 = DownACS(512, 1024 // factor, return_splits=True)

        if self.freeze_encoder is True:
            for p in self.inc.parameters():
                p.requires_grad = False
            for p in self.down1.parameters():
                p.requires_grad = False
            for p in self.down2.parameters():
                p.requires_grad = False
            for p in self.down3.parameters():
                p.requires_grad = False
            for p in self.down4.parameters():
                p.requires_grad = False
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes) if apply_sigmoid_to_output is False else OutConv(64, n_classes, sigmoid=True)

        if self.encoder_depth == 1:
            self.fc1 = nn.Linear(42, 3, bias=True)
        if self.encoder_depth == 2:
            self.fc1 = nn.Linear(85, 3, bias=True)
        if self.encoder_depth == 3:
            self.fc1 = nn.Linear(170, 3, bias=True)
        if self.encoder_depth == 4:
            if pool_features:
                # make 2,2,2 cube, in ACS they reduce to 1 but that seems too reductive
                self.fc1 = nn.Linear(170 * 2 * 2 * 2, 3, bias=True)
            else:
                self.fc1 = nn.Linear(170 * 4 * 4 * 2, 3, bias=True)

    def forward(self, x):

        # sanity check that weights dont update
        if self.freeze_encoder:

            if self.check_against is None:
                self.check_against_inc = self.inc.double_conv[3].weight.detach().clone()
                self.check_against_1 = self.down1.conv1.weight.detach().clone()
                self.check_against_2 = self.down2.conv2.weight.detach().clone()
                self.check_against_3 = self.down3.conv1.weight.detach().clone()
                self.check_against_4 = self.down4.conv2.weight.detach().clone()
                self.check_against = True

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            w0 = self.inc.double_conv[3].weight.detach().clone()
            w1 = self.down1.conv1.weight.detach().clone()
            w2 = self.down2.conv2.weight.detach().clone()
            w3 = self.down3.conv1.weight.detach().clone()
            w4 = self.down4.conv2.weight.detach().clone()

            self.check_against_inc, w0 = self.check_against_inc.to(device), w0.to(device)
            self.check_against_1, w1 = self.check_against_1.to(device), w1.to(device)
            self.check_against_2, w2 = self.check_against_2.to(device), w2.to(device)
            self.check_against_3, w3 = self.check_against_3.to(device), w3.to(device)
            self.check_against_4, w4 = self.check_against_4.to(device), w4.to(device)
            # assert weight is same
            assert torch.all(torch.eq(self.check_against_inc, w0))
            assert torch.all(torch.eq(self.check_against_1, w1))
            assert torch.all(torch.eq(self.check_against_2, w2))
            assert torch.all(torch.eq(self.check_against_3, w3))
            assert torch.all(torch.eq(self.check_against_4, w4))

        x1 = self.inc(x)
        x2, shapea_1, shapec_1, shapes_1 = self.down1(x1)
        # print(shapea_1, shapec_1, shapes_1)
        x3, shapea_2, shapec_2, shapes_2 = self.down2(x2)
        # print(shapea_2, shapec_2, shapes_2)
        x4, shapea_3, shapec_3, shapes_3 = self.down3(x3)
        # print(shapea_3, shapec_3, shapes_3)
        x5, shapea_4, shapec_4, shapes_4 = self.down4(x4)
        # print(shapea_4, shapec_4, shapes_4)

        # do the classificationa as well
        if self.encoder_depth == 4:
            x_cls, targets, _ = self._get_data_for_classification(last_down_features=x5, shape1=shapea_4, shape2=shapec_4, shape3=shapes_4)
            if self.pool_features:
                # reduce spatial dimensions and flatten them into (B,N)
                x_cls = F.adaptive_avg_pool3d(x_cls, output_size=2).view(x_cls.size(0), -1)
            else:
                x_cls = x_cls.view(x_cls.size(0), -1)  # (B,170,4,4,2) -> flatten along batch dimensions

        x_cls = self.fc1(x_cls)
        return (x_cls, targets)

    def _get_data_for_classification(self, last_down_features, shape1, shape2, shape3):

        x5 = last_down_features.detach().clone()  # (B,C,H,W,D)
        x5 = x5.to(self.device)
        batch_size = x5.shape[0]
        # equal split in relation to batch of how many examples from each view go to classifier
        # so from the elemensts of btach, how many featuers from each view will follow
        each_view = [batch_size // 3 for _ in range(3)]

        while sum(each_view) != batch_size:
            increment_choice = choice([0, 1, 2])
            each_view[increment_choice] += 1
        assert sum(each_view) == batch_size
        if 0 in each_view:
            print("EACH VIEW {}, because batch is of size {}".format(each_view, batch_size))

        # features belonging to each view
        x_a, x_c, x_s = x5[:, :shape1], x5[:, shape1 : shape1 + shape2], x5[:, shape1 + shape2 :]
        allowed_number_features = min(x_a.shape[1], x_c.shape[1], x_s.shape[1])  # due to kernel split we can have eg: 170,171,171

        res = []
        # take from each view's features the elements of batch that will follow to classification as each batch element has 3 views and only one will be used
        res.append(x_a[: each_view[0]])
        res.append(x_c[each_view[0] : each_view[0] + each_view[1]])
        res.append(x_s[each_view[0] + each_view[1] :])

        # make all have same features nr as in allowed_number_features
        for idx, i in enumerate(res):
            if i.shape[1] != allowed_number_features:
                i = torch.narrow(i, 1, 1, i.shape[1] - 1)
                res[idx] = i

        targets = torch.LongTensor([0 for i in range(each_view[0])] + [1 for i in range(each_view[1])] + [2 for i in range(each_view[2])])
        res_out = torch.cat(res, dim=0)  # concatenate on batch
        # print(res_out.shape)
        return res_out, targets, allowed_number_features


class UnetACSAxisAwareDecoder(nn.Module):
    "To be used with ACS conversion only"

    def __init__(self, n_channels, n_classes, bilinear=True, apply_sigmoid_to_output=False):
        super(UnetACSAxisAwareDecoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.apply_sigmoid_to_output = apply_sigmoid_to_output

        self.inc = DoubleConvACS(n_channels, 64)
        self.down1 = DownACS(64, 128, return_splits=True)
        self.down2 = DownACS(128, 256, return_splits=True)
        self.down3 = DownACS(256, 512, return_splits=True)
        self.factor = 2 if bilinear else 1
        self.down4 = DownACS(512, 1024 // self.factor, return_splits=True)

        self.up1 = AxisAwareUpBlock((342, 342, 340), 512 // self.factor, bilinear=self.bilinear)
        self.up1.to(self.device)
        self.up2 = AxisAwareUpBlock((172, 170, 170), 256 // self.factor, bilinear=self.bilinear)
        self.up2.to(self.device)
        self.up3 = AxisAwareUpBlock((86, 86, 84), 128 // self.factor, bilinear=self.bilinear)
        self.up3.to(self.device)
        self.up4 = AxisAwareUpBlock((44, 42, 42), 64, bilinear=self.bilinear)
        self.up4.to(self.device)
        self.outc = OutConv(64, self.n_classes) if self.apply_sigmoid_to_output is False else OutConv(64, self.n_classes, sigmoid=True)

        self.outc.to(self.device)

    def forward(self, x):

        x1, shape1_1, shape2_1, shape3_1 = self.inc(x)

        x2, shape1_2, shape2_2, shape3_2 = self.down1(x1)
        x3, shape1_3, shape2_3, shape3_3 = self.down2(x2)
        x4, shape1_4, shape2_4, shape3_4 = self.down3(x3)
        x5, shape1_5, shape2_5, shape3_5 = self.down4(x4)

        x1_a, x1_c, x1_s = x1[:, :shape1_1], x1[:, shape1_1 : shape1_1 + shape2_1], x1[:, shape1_1 + shape2_1 :]
        x2_a, x2_c, x2_s = x2[:, :shape1_2], x2[:, shape1_2 : shape1_2 + shape2_2], x2[:, shape1_2 + shape2_2 :]
        x3_a, x3_c, x3_s = x3[:, :shape1_3], x3[:, shape1_3 : shape1_3 + shape2_3], x3[:, shape1_3 + shape2_3 :]
        x4_a, x4_c, x4_s = x4[:, :shape1_4], x4[:, shape1_4 : shape1_4 + shape2_4], x4[:, shape1_4 + shape2_4 :]
        x5_a, x5_c, x5_s = x5[:, :shape1_5], x5[:, shape1_5 : shape1_5 + shape2_5], x5[:, shape1_5 + shape2_5 :]

        x = self.up1((x5_a, x4_a, x5_c, x4_c, x5_s, x4_s))
        x, shape1, shape2, shape3 = x
        x_a, x_c, x_s = x[:, :shape1], x[:, shape1 : shape1 + shape2], x[:, shape1 + shape2 :]
        # print("PRE UP 2: {} {} {}, OUT FROM UP 1 was {}".format(x_a.shape, x_c.shape, x_s.shape, x.shape))

        x = self.up2((x_a, x3_a, x_c, x3_c, x_s, x3_s))
        x, shape1, shape2, shape3 = x
        x_a, x_c, x_s = x[:, :shape1], x[:, shape1 : shape1 + shape2], x[:, shape1 + shape2 :]

        x = self.up3((x_a, x2_a, x_c, x2_c, x_s, x2_s))
        x, shape1, shape2, shape3 = x
        x_a, x_c, x_s = x[:, :shape1], x[:, shape1 : shape1 + shape2], x[:, shape1 + shape2 :]
        # print("PRE UP 4: {} {} {}, OUT FROM UP 3 was {}".format(x_a.shape, x_c.shape, x_s.shape, x.shape))
        x, shape1, shape2, shape3 = self.up4((x_a, x1_a, x_c, x1_c, x_s, x1_s))
        logits = self.outc(x)
        # print("SHAPE LOGITS", logits.shape)
        return logits


if __name__ == "__main__":
    import torch

    from ACSConv.acsconv.converters import ACSConverter

    r2d = torch.randn(1, 1, 64, 64)
    r3d = torch.randn(5, 1, 16, 16, 16)
    u_2d = UNet(1, 1)
    # u_2d(r2d)
    # u_2d_acs = ACSConverter(u_2d)
    # u_2d_acs(r3d)

    u = UnetACSWithClassifier(
        1,
        1,
    )
    # u = UnetACSAxisAwareDecoder(1, 1)
    u_acs = ACSConverter(u)
    u_acs(r3d)