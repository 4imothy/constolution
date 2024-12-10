import os
import sys
import torch
import constolution as ct
from torch import nn
from constolution.pd_kernels import to_tensor, Kernels


class earlyBlock(nn.Module):

    def __init__(self, input_channels, output_channels, stride, spatial_size):
        super(earlyBlock, self).__init__()
        
        # early pre-defined filters matrix operation on Gabor
        self.filter1 = ct.Constolution2D(ct.Kernels.Gabor, 
        input_channels, output_channels, stride=stride, spatial_size=spatial_size)

        self.filter2 = ct.Constolution2D(ct.Kernels.SobelHorizontalEdge, 
        input_channels, output_channels, stride=stride, spatial_size=spatial_size) # DECIDE HORIZONTAL VS VERT

        self.filter3 = ct.Constolution2D(ct.Kernels.Schmid, 
        input_channels, output_channels, stride=stride, spatial_size=spatial_size)

        self.filter4 = ct.Constolution2D(ct.Constolution2D.Gaussian, 
        input_channels, output_channels, stride=stride, spatial_size=spatial_size)
        # FIFTH KERNEL TBD

    "gabor, sobel, schmidit, gaussian, *five_one"

    """ORDERING: FLEXILITY, FORWARD FUNCTION, TAKES IN LIST, THEN DEFINE WHATS 
    IN THE LIST, COMBINATION WITHIN LIST"""

    """Instaniating filters later on, not setting filters"""

    """WE MIGHT NEED TO FIGURE IF WE NEED TO PASS it in"""
    def forward(self, x):
        pass


class middleBlock(nn.Module):

    def __init__(self, input_channels, output_channels, stride):
        super(earlyBlock, self).__init__()
        "kernel1, kernel2, kernel3, kernel4, kernel5"
        self.filter1 = ct.Constolution2D(ct.Kernels.Gabor, 
            input_channels, output_channels, stride=stride)
        
        self.filter2 = ct.Constolution2D(ct.Kernels.Gabor, 
            input_channels, output_channels, stride=stride)
        
        self.filter3 = ct.Constolution2D(ct.Kernels.Gabor, 
            input_channels, output_channels, stride=stride)
        
        self.filter4 = ct.Constolution2D(ct.Kernels.Gabor, 
            input_channels, output_channels, stride=stride)


    def forward():
        pass

class EarlyBlockFeatureMapWeighted(nn.Module):
    def __init__(self, input_channels, output_channels, stride, spatial_size):
        super(EarlyBlockFeatureMapWeighted, self).__init__()

        self.filter1 = ct.Constolution2D(ct.Kernels.Gabor,input_channels, output_channels, stride=stride, spatial_size=(3, 3))
        self.filter2 = ct.Constolution2D(ct.Kernels.Gaussian,input_channels, output_channels, stride=stride, spatial_size=(3, 3))
        self.filter3 = ct.Constolution2D(ct.Kernels.Identity,input_channels, output_channels, stride=stride, spatial_size=(3, 3))
        #self.filter3 = ct.Constolution2D(ct.Kernels.SobelHorizontalEdge,input_channels, output_channels, stride=stride, spatial_size=(3, 3))
        #self.filter4 = ct.Constolution2D(ct.Kernels.SobelVerticalEdge,input_channels, output_channels, stride=stride, spatial_size=(3, 3))
        #self.filter5 = ct.Constolution2D(ct.Kernels.Schmid,input_channels, output_channels, stride=stride, spatial_size=(3, 3))
        self.scalar_weights = nn.Parameter(torch.ones(5))
        #self.conv = nn.Conv2d(
        #    in_channels=output_channels,
        #    out_channels=output_channels,
        #    kernel_size=3,
        #    stride=1,
        #    padding=1
        #)

    def forward(self, x):
        fmap1 = self.filter1(x)
        fmap2 = self.filter2(x)
        fmap3 = self.filter3(x)
        #fmap3 = self.filter3(x)
        #fmap4 = self.filter4(x)
        #fmap5 = self.filter5(x)
        feature_maps = torch.stack([fmap1, fmap2, fmap3], dim=1) 
        #feature_maps = torch.stack([fmap1, fmap2, fmap3, fmap4, fmap5], dim=1) 
        weighted_feature_maps = feature_maps * self.scalar_weights.view(1, -1, 1, 1, 1)
        out = weighted_feature_maps.sum(dim=1)
        #out = self.conv(out)

        return out