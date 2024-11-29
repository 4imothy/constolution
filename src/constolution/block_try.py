import os
import sys
import torch
import constolution as ct
from torch import nn


class earlyBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):

        super(earlyBlock, self).__init__()
        
        kernel_list = [
            ct.Kernels.Gaussian, 
            ct.Kernels.VerticalEdge, 
            ct.Kernels.HorizontalEdge, 
            ct.Kernels.Box,
            ct.Kernels.SobelVerticalEdge
        ]
        self.filters = nn.ModuleList([
            ct.Constolution2D(kernel, input_channels, output_channels, stride=stride)
            for kernel in kernel_list
        ])
    
    def forward(self, x):
        filter_outputs = [filter_layer(x) for filter_layer in self.filters]        
        combined_output = torch.cat(filter_outputs, dim=1)
        return self.combine(combined_output)


class middleBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):

        super(MiddleBlock, self).__init__()
        
        kernel_list = [
            ct.Kernels.Gabor, 
            ct.Kernels.Schmid, 
            ct.Kernels.SobelHorizontalEdge, 
            ct.Kernels.SobelVerticalEdge,
            ct.Kernels.random_basis_gaussian_sparse
        ]
        self.filters = nn.ModuleList([
            ct.Constolution2D(kernel, input_channels, output_channels, stride=stride)
            for kernel in kernel_list
        ])
    
    def forward(self, x):
        filter_outputs = [filter_layer(x) for filter_layer in self.filters]        
        combined_output = torch.cat(filter_outputs, dim=1)
        return self.combine(combined_output)