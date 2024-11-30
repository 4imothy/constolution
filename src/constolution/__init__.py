from typing import Tuple, Union, Sequence
import torch
from torch import nn
from . import pd_kernels
import torch.nn.functional as F

Kernels = pd_kernels.Kernels

class Constolution2D(nn.Module):
    def __init__(self, type: Kernels, in_channels: int, out_channels: int,
                spatial_size: Union[int, Tuple[int, int]], stride=1,
                padding=1, dilation=1, groups=1, bias=True, depthwise=False, **kwargs):
    
        super(Constolution2D, self).__init__()

        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)

        if depthwise:
            assert out_channels % in_channels == 0, "out_channels must be divisible by in_channels for depthwise convolution"
            groups = in_channels
            
        self.weight = nn.Parameter(
            pd_kernels.to_tensor(type, in_channels, out_channels, spatial_size, groups, **kwargs),
            requires_grad=False
        )

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None
            
    def forward(self, x: torch.Tensor):
        self.weight = self.weight.to(x.device)
        if self.bias is not None:
            self.bias = self.bias.to(x.device)
        
        return torch.nn.functional.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

class BaseInceptionConstolution2D(nn.Module):
    def __init__(self, operators: Sequence[Union[nn.Sequential, Constolution2D]],
                 total_out_channels: int,
                 weighted_convolution=True):
        self.operators = operators
        self.weight = None
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

class EdgeInception(BaseInceptionConstolution2D):
    def __init__(self, in_channels: int, out_channels: int,
                 spatial_size: Union[int, Tuple[int, int]], stride=1,
                 padding=1, dilation=1, groups=1, bias=True, depthwise=False):
        operators = [Constolution2D(kern, in_channels, out_channels,
                                    spatial_size, stride, padding, dilation,
                                    groups, bias, depthwise) for kern in
                     [Kernels.VerticalEdge, Kernels.HorizontalEdge,
                      Kernels.SobelVerticalEdge, Kernels.SobelHorizontalEdge]]
        super().__init__(operators, out_channels * len(operators))


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