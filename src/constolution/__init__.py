from typing import Tuple, Union, Sequence, List, Optional
import torch
from torch import nn
from . import pd_kernels
import torch.nn.functional as F

Kernels = pd_kernels.Kernels

class Constolution2D(nn.Module):

    # TODO need to take the hparams from user (sigma)
    def __init__(self, type: Kernels, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]], stride=1,
                 padding=1, dilation=1, groups=1, bias=True, depthwise=False, **kwargs):
        super(Constolution2D, self).__init__()
        if depthwise:
            assert out_channels % in_channels == 0
            groups = in_channels

        self.weight = nn.Parameter(pd_kernels.to_tensor(type, in_channels,
                                                        out_channels,
                                                        kernel_size, groups,
                                                        **kwargs),
                                   requires_grad=False)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class BaseInceptionConstolution2D(nn.Module):
    def __init__(self, operators: Sequence[Union[nn.Sequential, Constolution2D]],
                 total_out_channels: int,
                 out_channels: int,
                 weighted_convolution=True):
        super().__init__()
        self.operators = nn.ModuleList(operators)
        self.weight = None
        if weighted_convolution:
            self.weight = nn.Conv2d(total_out_channels, out_channels, (1,1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batched = x.dim() == 4
        outs = [op(x) for op in self.operators]
        cat = torch.cat(outs, dim=1 if batched else 0)
        if self.weight is not None:
            return self.weight(cat)
        return cat

class BaseInceptionSharedHParams(BaseInceptionConstolution2D):
    KERNELS: List[Kernels] = []
    def __init__(self, in_channels: int, out_channels: int, kernel_size:
                 Union[int, Tuple[int, int]], stride=1, padding=1, dilation=1,
                 groups=1, bias=True, depthwise=False, weighted_out_channels:
                 Optional[int] = None, weighted_convolution=True):
        operators = [Constolution2D(kern, in_channels, out_channels,
                                    kernel_size, stride, padding, dilation,
                                    groups, bias, depthwise) for kern in
                     self.KERNELS]
        total_out_channels = out_channels * len(operators)
        weighted_out_channels = weighted_out_channels or total_out_channels
        super().__init__(operators, total_out_channels, weighted_out_channels, weighted_convolution=weighted_convolution)

class EdgeInception(BaseInceptionSharedHParams):
    KERNELS = [
            Kernels.VerticalEdge, Kernels.HorizontalEdge,
            Kernels.SobelVerticalEdge, Kernels.SobelHorizontalEdge
            ]

class EarlyEdgeInception(BaseInceptionSharedHParams):
    KERNELS = [Kernels.Gabor, Kernels.Schmid,
               Kernels.SobelVerticalEdge, Kernels.SobelHorizontalEdge,
               Kernels.Gaussian]

class MiddleInception(BaseInceptionSharedHParams):
    KERNELS = [Kernels.Gaussian,
               Kernels.Box,
               Kernels.Identity,
               Kernels.Average,
               ]

class earlyBlock(nn.Module):

    def __init__(self, input_channels, output_channels, stride):
        super(earlyBlock, self).__init__()

        # early pre-defined filters matrix operation on Gabor
        self.filter1 = Constolution2D(Kernels.Gabor,
        input_channels, output_channels, stride=stride)

        self.filter2 = Constolution2D(Kernels.SobelHorizontalEdge,
        input_channels, output_channels, stride=stride) # DECIDE HORIZONTAL VS VERT

        self.filter3 = Constolution2D(Kernels.Schmid,
        input_channels, output_channels, stride=stride)

        self.filter4 = Constolution2D(Constolution2D.Gaussian,
        input_channels, output_channels, stride=stride)
        # FIFTH KERNEL TBD

    "gabor, sobel, schmidit, gaussian, *five_one"

    """ORDERING: FLEXILITY, FORWARD FUNCTION, TAKES IN LIST, THEN DEFINE WHATS
    IN THE LIST, COMBINATION WITHIN LIST"""

    """Instaniating filters later on, not setting filters"""

    """WE MIGHT NEED TO FIGURE IF WE NEED TO PASS it in"""
    def forward(self, x):
        pass


class sillyBlock(nn.Module):

    def init(self, input_channels, output_channels, stride):
        super(sillyBlock, self).init()
        "kernel1, kernel2, kernel3, kernel4, kernel5"
        self.filter1 = Constolution2D(Kernels.Gabor,
            input_channels, output_channels, stride=stride)

        self.filter2 = Constolution2D(Kernels.Gabor,
            input_channels, output_channels, stride=stride)

        self.filter3 = Constolution2D(Kernels.Gabor,
            input_channels, output_channels, stride=stride)

        self.filter4 = Constolution2D(Kernels.Gabor,
            input_channels, output_channels, stride=stride)

        self.kernels = nn.ModuleList([self.filter1, self.filter2,self.filter3,self.filter4])
        self.alpha = nn.Parameter(torch.ones(len(self.kernels)) / len(self.kernels))

    def forward(self, x):
        outputs = [kernel(x) for kernel in self.kernels]
        weighted_output = sum(F.softmax(self.alpha, dim=0)[i] * outputs[i] for i in range(len(outputs)))
        return weighted_output
