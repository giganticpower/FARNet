import torch.nn as nn
import torch.nn.functional as F

import math


class Conv_BN_ReLU(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPEM_v2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPEM_v2, self).__init__()
        planes = out_channels
        self.dwconv4_1 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer4_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv3_1 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_1 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv1_1 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer2_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv3_2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer3_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv4_2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer4_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv5_2 = nn.Conv2d(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=planes,
                                   bias=False)
        self.smooth_layer5_2 = Conv_BN_ReLU(planes, planes)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, f1, f2, f3, f4, f5):
        f4_ = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f5, f4)))
        f3_ = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4_, f3)))
        f2_ = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3_, f2)))
        f1_ = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2_, f1)))

        # f2_ = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2_,
        #                                                              f1_)))
        # f3_ = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3_,
        #                                                              f2_)))
        # f4_ = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4_, f3_)))
        #
        # f5_ = self.smooth_layer5_2(self.dwconv5_2(self._upsample_add(f5, f4_)))

        # f1 = f1 + f1_
        # f2 = f2 + f2_
        # f3 = f3 + f3_
        # f4 = f4 + f4_
        # f5 = f5 + f5_

        return f1_, f2_, f3_, f4_, f5
        #return f1, f2, f3, f4, f5
