import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet as EffNet
from network.nets.layers import (Conv2dStaticSamePadding, MaxPool2dStaticSamePadding,
                         MemoryEfficientSwish, Swish)


# ----------------------------------#
#   Xception中深度可分离卷积
#   先3x3的深度可分离卷积
#   再1x1的普通卷积
# ----------------------------------#
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class UpBlok_1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x


class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time

        if self.first_time:
            ## cc adding:
            self.p7_down_channels = nn.Sequential(
                Conv2dStaticSamePadding(320, 64, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p6_down_channels = nn.Sequential(
                Conv2dStaticSamePadding(112, 64, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channels = nn.Sequential(
                Conv2dStaticSamePadding(40, 64, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channels = nn.Sequential(
                Conv2dStaticSamePadding(24, 64, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channels = nn.Sequential(
                Conv2dStaticSamePadding(16, 64, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        #         简易注意力机制的weights
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """ bifpn模块结构示意图
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """
        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):
        # ------------------------------------------------#
        #   当phi=1、2、3、4、5的时候使用fast_attention
        #   获得三个shape的有效特征层
        #   分别是C3  64, 64, 40
        #         C4  32, 32, 112
        #         C5  16, 16, 320
        # ------------------------------------------------#
        if self.first_time:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            p7_in = self.p7_down_channels(p7_in)
            p6_in = self.p6_down_channels(p6_in)
            p5_in = self.p5_down_channels(p5_in)
            p4_in = self.p4_down_channels(p4_in)
            p3_in = self.p3_down_channels(p3_in)

            # 简单的注意力机制，用于确定更关注p7_in还是p6_in
            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            # 简单的注意力机制，用于确定更关注p6_up还是p5_in
            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_td)))

            # 简单的注意力机制，用于确定更关注p5_up还是p4_in
            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_td)))

            # 简单的注意力机制，用于确定更关注p4_up还是p3_in
            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

            # 简单的注意力机制，用于确定更关注p4_in还是p4_up还是p3_out
            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))

            # 简单的注意力机制，用于确定更关注p5_in还是p5_up还是p4_out
            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out)))

            # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * p6_td + weight[2] * self.p6_downsample(p5_out)))

            # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            # 简单的注意力机制，用于确定更关注p7_in还是p6_in
            p6_w1 = self.p6_w1_relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

            # 简单的注意力机制，用于确定更关注p6_up还是p5_in
            p5_w1 = self.p5_w1_relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_td)))

            # 简单的注意力机制，用于确定更关注p5_up还是p4_in
            p4_w1 = self.p4_w1_relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_td)))

            # 简单的注意力机制，用于确定更关注p4_up还是p3_in
            p3_w1 = self.p3_w1_relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

            # 简单的注意力机制，用于确定更关注p4_in还是p4_up还是p3_out
            p4_w2 = self.p4_w2_relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.conv4_down(
                self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))

            # 简单的注意力机制，用于确定更关注p5_in还是p5_up还是p4_out
            p5_w2 = self.p5_w2_relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.conv5_down(
                self.swish(weight[0] * p5_in + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out)))

            # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
            p6_w2 = self.p6_w2_relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.conv6_down(
                self.swish(weight[0] * p6_in + weight[1] * p6_td + weight[2] * self.p6_downsample(p5_out)))

            # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
            p7_w2 = self.p7_w2_relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out


    def _forward(self, inputs):
        # 当phi=6、7的时候使用_forward
        if self.first_time:
            # 第一次BIFPN需要下采样与降通道获得
            # p3_in p4_in p5_in p6_in p7_in
            p3, p4, p5 = inputs
            p3_in = self.p3_down_channel(p3)
            p4_in_1 = self.p4_down_channel(p4)
            p4_in_2 = self.p4_down_channel_2(p4)
            p5_in_1 = self.p5_down_channel(p5)
            p5_in_2 = self.p5_down_channel_2(p5)
            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p6_td = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

            p5_td = self.conv5_up(self.swish(p5_in_1 + self.p5_upsample(p6_td)))

            p4_td = self.conv4_up(self.swish(p4_in_1 + self.p4_upsample(p5_td)))

            p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td)))

            p4_out = self.conv4_down(
                self.swish(p4_in_2 + p4_td + self.p4_downsample(p3_out)))

            p5_out = self.conv5_down(
                self.swish(p5_in_2 + p5_td + self.p5_downsample(p4_out)))

            p6_out = self.conv6_down(
                self.swish(p6_in + p6_td + self.p6_downsample(p5_out)))

            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            p6_td = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

            p5_td = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_td)))

            p4_td = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_td)))

            p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td)))

            p4_out = self.conv4_down(
                self.swish(p4_in + p4_td + self.p4_downsample(p3_out)))

            p5_out = self.conv5_down(
                self.swish(p5_in + p5_td + self.p5_downsample(p4_out)))

            p6_out = self.conv6_down(
                self.swish(p6_in + p6_td + self.p6_downsample(p5_out)))

            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class EfficientNet(nn.Module):
    def __init__(self, phi):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{phi}')
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            # ------------------------------------------------------#
            #   取出对应的特征层，如果某个EffcientBlock的步长为2的话
            #   意味着它的前一个特征层为有效特征层
            #   除此之外，最后一个EffcientBlock的输出为有效特征层
            # ------------------------------------------------------#
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[0:]


class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=8, phi=0):
        super(EfficientDetBackbone, self).__init__()
        # --------------------------------#
        #   phi指的是efficientdet的版本
        # --------------------------------#
        self.phi = phi
        # ---------------------------------------------------#
        #   backbone_phi指的是该efficientdet对应的efficient
        # ---------------------------------------------------#
        self.backbone_phi = [0, 1, 2, 3, 4, 5, 6, 6]
        # --------------------------------#
        #   BiFPN所用的通道数
        # --------------------------------#
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        # --------------------------------#
        #   BiFPN的重复次数
        # --------------------------------#
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        # ---------------------------------------------------#
        #   Effcient Head卷积重复次数
        # ---------------------------------------------------#
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        # ---------------------------------------------------#
        #   基础的先验框大小
        # ---------------------------------------------------#
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]

        self.check_first_time = [True, False, False]

        conv_channel_coef = {
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.phi],
                    conv_channel_coef[phi],
                    self.check_first_time[_],
                    attention=True if phi < 6 else False)
              for _ in range(self.fpn_cell_repeats[phi])])

        self.num_classes = num_classes

        self.backbone_net = EfficientNet(self.backbone_phi[phi])

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    # def forward(self, inputs):
    #     p1, p2, p3, p4, p5 = self.backbone_net(inputs)
    #
    #     features = (p1, p2, p3, p4, p5)
    #     n1, n2, n3, n4, n5 = self.bifpn(features)
    #     p3_end = torch.cat((p1, n1), dim=1)
    #     p4_end = torch.cat((p2, n2), dim=1)
    #     p5_end = torch.cat((p3, n3), dim=1)
    #     p6_end = torch.cat((p4, n4), dim=1)
    #     p7_end = torch.cat((p5, n5), dim=1)
    #
    #     return p3_end, p4_end, p5_end, p6_end, p7_end
    def forward(self, inputs):
        p1, p2, p3, p4, p5 = self.backbone_net(inputs)

        features = (p1, p2, p3, p4, p5)
        n1, n2, n3, n4, n5 = self.bifpn(features)

        return n1, n2, n3, n4, n5


if __name__ == '__main__':
    input = torch.rand(1, 3, 640, 640)
    net = EfficientDetBackbone()

    n1,n2,n3,n4,n5 = net(input)

    print(n1.size())
    print(n2.size())
    print(n3.size())
    print(n4.size())
    print(n5.size())
