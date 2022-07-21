import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:input features with shape [b,c,h,w]
        b, c, h, w  = x.size()

        #feature descriptor on the global spatial information
        y = self.avg_pool(x) + self.max_pool(x)
        #y = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=0)
        #y = self.avg_pool(x)
        #y = self.max_pool(x)


        #Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        #print(y.shape)
        #Multi-scale information fusion
        y = self.sigmoid(y)
        #print(y.shape)
        #return x * y.expand_as(x)
        return x * y.expand_as(x)

class esa_layer(nn.Module):
    """Constructs a ESA module.

        Args:
            Spatial: the input feature map
            k_size: Adaptive selection of kernel size
        """

    def __init__(self, planes, kernel_size=7):
        super(esa_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(planes, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:input features with shape [b,c,h,w]
        b, c, h, w = x.size()
        #avgout = torch.mean(x, dim=1, keepdim=True)
        #maxout, _ = torch.max(x, dim=1, keepdim=True)

        #x = torch.cat([avgout, maxout], dim=1)
        # feature descriptor on the global spatial information
        #y = torch.cat([self.avg_pool(x),self.max_pool(x)],dim=0)
        y = self.avg_pool(x) + self.max_pool(x)
        # y = self.avg_pool(x)
        #y = self.max_pool(x)

        # Two different branches of ECA module
        #y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.conv(y)

        # Multi-scale information fusion
        y = self.sigmoid(y)


        return x * y

