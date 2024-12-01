# import os
# import sys
# import torch
# import constolution as ct
# from torch import nn


# class EarlyBlock(nn.Module):
#     def __init__(self, input_channels, output_channels, stride, spatial_size):

#         super(EarlyBlock, self).__init__()
        
#         kernel_list = [
#             ct.Kernels.Gabor, 
#             ct.Kernels.Schmid, 
#             ct.Kernels.SobelHorizontalEdge, 
#             ct.Kernels.SobelVerticalEdge,
#             ct.Kernels.Gaussian
#         ]
        
#         self.filters = nn.ModuleList([
#             ct.Constolution2D(kernel, input_channels, output_channels, stride=stride, spatial_size=spatial_size)
#             for kernel in kernel_list
#         ])
        
#         self.combine = nn.Conv2d(
#             output_channels * len(kernel_list),
#             output_channels,
#             kernel_size=1
#         )
    
#     def forward(self, x):
#         filter_outputs = [filter_layer(x) for filter_layer in self.filters]        
#         combined_output = torch.cat(filter_outputs, dim=1)
#         return self.combine(combined_output)


# class MiddleBlock(nn.Module):
#     def __init__(self, input_channels, output_channels, stride, spatial_size):

#         super(MiddleBlock, self).__init__()
        
#         kernel_list = [
#             ct.Kernels.Gaussian, 
#             ct.Kernels.VerticalEdge, 
#             ct.Kernels.HorizontalEdge, 
#             ct.Kernels.Box,
#             ct.Kernels.SobelVerticalEdge
#         ]
#         self.filters = nn.ModuleList([
#             ct.Constolution2D(kernel, input_channels, output_channels, stride=stride, spatial_size=spatial_size)
#             for kernel in kernel_list
#         ])
        
#         self.combine = nn.Conv2d(
#             output_channels * len(kernel_list),
#             output_channels,
#             kernel_size=1
#         )
    
#     def forward(self, x):
#         filter_outputs = [filter_layer(x) for filter_layer in self.filters]        
#         combined_output = torch.cat(filter_outputs, dim=1)
#         return self.combine(combined_output)

import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride, spatial_size):
        super(EarlyBlock, self).__init__()
        
        kernel_list = [
            ct.Kernels.Gabor, 
            ct.Kernels.Schmid, 
            ct.Kernels.SobelHorizontalEdge, 
            ct.Kernels.SobelVerticalEdge,
            ct.Kernels.Gaussian
        ]
        
        num_kernels = len(kernel_list)
        kernel_size = kernel_list[0].shape[-1]
        
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels * num_kernels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=input_channels,
            bias=False
        )
        
        with torch.no_grad():
            weight = self.conv.weight 
            kernels = torch.stack(kernel_list)
            kernels = kernels.unsqueeze(1)
            kernels = kernels.repeat(1, output_channels, 1, 1)
            kernels = kernels.view(-1, 1, kernel_size, kernel_size)
            weight.copy_(kernels)
        
        self.conv.weight.requires_grad = False
        
        self.combine = nn.Conv2d(
            in_channels=output_channels * num_kernels,
            out_channels=output_channels,
            kernel_size=1,
            bias=False
        )
        
    def forward(self, x):
        out = self.conv(x)
        out = self.combine(out)
        return out

class MiddleBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride, spatial_size):
        super(MiddleBlock, self).__init__()
        
        kernel_list = [
            ct.Kernels.Gaussian, 
            ct.Kernels.VerticalEdge, 
            ct.Kernels.HorizontalEdge, 
            ct.Kernels.Box,
            ct.Kernels.SobelVerticalEdge
        ]
        
        num_kernels = len(kernel_list)
        kernel_size = kernel_list[0].shape[-1]
        
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels * num_kernels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=input_channels,
            bias=False
        )
        
        with torch.no_grad():
            weight = self.conv.weight
            kernels = torch.stack(kernel_list)
            kernels = kernels.unsqueeze(1)
            kernels = kernels.repeat(1, output_channels, 1, 1)
            kernels = kernels.view(-1, 1, kernel_size, kernel_size)
            weight.copy_(kernels)
        
        self.conv.weight.requires_grad = False
        
        self.combine = nn.Conv2d(
            in_channels=output_channels * num_kernels,
            out_channels=output_channels,
            kernel_size=1,
            bias=False
        )
        
    def forward(self, x):
        out = self.conv(x)
        out = self.combine(out)
        return out
