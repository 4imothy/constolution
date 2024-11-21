from typing import Tuple, Union
import torch
from torch import nn
from . import pd_kernels

Kernels = pd_kernels.Kernels

class Constolution2D(nn.Module):

    # TODO need to take the hparams from user (sigma)
    def __init__(self, type: Kernels, in_channels: int, out_channels: int,
                 spatial_size: Union[int, Tuple[int, int]], stride=1,
                 padding=1, dilation=1, groups=1, bias=True):
        super(Constolution2D, self).__init__()
        self.weight = nn.Parameter(pd_kernels.to_tensor(type, in_channels, out_channels, spatial_size, groups), requires_grad=False)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = torch.empty(out_channels)
        else:
            bias = None

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

