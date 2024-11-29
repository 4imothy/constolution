import os
import sys
import torch
import constolution as ct
from torch import nn


class EarlyBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride, spatial_size):

        super(EarlyBlock, self).__init__()
        
        kernel_list = [
            ct.Kernels.Gaussian, 
            ct.Kernels.VerticalEdge, 
            ct.Kernels.HorizontalEdge, 
            ct.Kernels.Box,
            ct.Kernels.SobelVerticalEdge
        ]
        self.filters = nn.ModuleList([
            ct.Constolution2D(kernel, input_channels, output_channels, stride=stride, spatial_size=spatial_size)
            for kernel in kernel_list
        ])
    
    def forward(self, x):
        filter_outputs = [filter_layer(x) for filter_layer in self.filters]        
        combined_output = torch.cat(filter_outputs, dim=1)
        return self.combine(combined_output)


class MiddleBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride, spatial_size):

        super(MiddleBlock, self).__init__()
        
        kernel_list = [
            ct.Kernels.Gabor, 
            ct.Kernels.Schmid, 
            ct.Kernels.SobelHorizontalEdge, 
            ct.Kernels.SobelVerticalEdge,
            ct.Kernels.Gaussian
        ]
        self.filters = nn.ModuleList([
            ct.Constolution2D(kernel, input_channels, output_channels, stride=stride, spatial_size=spatial_size)
            for kernel in kernel_list
        ])
    
    def forward(self, x):
        filter_outputs = [filter_layer(x) for filter_layer in self.filters]        
        combined_output = torch.cat(filter_outputs, dim=1)
        return self.combine(combined_output)
