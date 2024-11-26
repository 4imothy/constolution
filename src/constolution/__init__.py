from typing import Tuple, Union, Sequence
import torch
from torch import nn
from . import pd_kernels

Kernels = pd_kernels.Kernels

class Constolution2D(nn.Module):

    # TODO need to take the hparams from user (sigma)
    def __init__(self, type: Kernels, in_channels: int, out_channels: int,
                 spatial_size: Union[int, Tuple[int, int]], stride=1,
                 padding=1, dilation=1, groups=1, bias=True, depthwise=False, **kwargs):
        super(Constolution2D, self).__init__()
        if depthwise:
            assert out_channels % in_channels == 0
            groups = in_channels

        self.weight = nn.Parameter(pd_kernels.to_tensor(type, in_channels,
                                                        out_channels,
                                                        spatial_size, groups,
                                                        **kwargs),
                                   requires_grad=False)
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

class BaseInceptionConstolution2D(nn.Module):
    def __init__(self, operators: Sequence[Union[nn.Sequential, Constolution2D]],
                 out_channels_of_paths: Sequence[int],
                 weighted_convolution=True, ):
        self.operators = operators
        self.weight = None
        total_out_channels = 0
        for oc in out_channels_of_paths:
            total_out_channels += oc
        if weighted_convolution:
            self.weight = nn.Conv2d(total_out_channels, 1, (1,1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batched = x.dim() == 4
        outs = []
        for op in self.operators:
            outs.append(op(x))
        if self.weight is not None:
            return self.weight(torch.cat(outs, dim=1 if batched else 0))
        return torch.cat(outs, dim=1 if batched else 0)
