import os
import sys
import torch
import constolution as ct
from torch import nn


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
        
        kernels = []
        for kernel_type in kernel_list:
            kernel = ct.to_tensor(
                type=kernel_type,
                in_channels=input_channels,
                out_channels=1,
                spatial_size=spatial_size,
                groups=1 
            )
            kernels.append(kernel)
        
        kernel_size = kernels[0].shape[-1]
        num_kernels = len(kernels)
        
        kernels = torch.stack(kernels, dim=0).squeeze(1)
        kernels = kernels.repeat(1, input_channels, 1, 1)
        
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels * num_kernels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=1,
            bias=False
        )
        
        with torch.no_grad():
            weight = kernels.repeat(output_channels, 1, 1, 1)
            self.conv.weight.copy_(weight)
        
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
        
        kernels = []
        for kernel_type in kernel_list:
            kernel = ct.to_tensor(
                type=kernel_type,
                in_channels=input_channels,
                out_channels=1,
                spatial_size=spatial_size,
                groups=1
            )
            kernels.append(kernel)
        
        kernel_size = kernels[0].shape[-1]
        num_kernels = len(kernels)
        
        kernels = torch.stack(kernels, dim=0).squeeze(1)
        kernels = kernels.repeat(1, input_channels, 1, 1)
        
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels * num_kernels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=1,
            bias=False
        )
        
        with torch.no_grad():
            weight = kernels.repeat(output_channels, 1, 1, 1)
            self.conv.weight.copy_(weight)
        
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
