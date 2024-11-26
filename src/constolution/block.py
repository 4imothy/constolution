import os
import sys
import torch
import constolution as ct
from torch import nn


class earlyBlock(nn.Module):

    def __init__(self, input_channels, output_channels, stride):
        super(earlyBlock, self).__init__()
        
        # early pre-defined filters matrix operation on Gabor
        self.filter1 = ct.Constolution2D(ct.Kernels.Gabor, 
        input_channels, output_channels, stride=stride)

        self.filter2 = ct.Constolution2D(ct.Kernels.SobelHorizontalEdge, 
        input_channels, output_channels, stride=stride) # DECIDE HORIZONTAL VS VERT

        self.filter3 = ct.Constolution2D(ct.Kernels.Schmid, 
        input_channels, output_channels, stride=stride)

        self.filter4 = ct.Constolution2D(ct.Constolution2D.Gaussian, 
        input_channels, output_channels, stride=stride
        )
        # FIFTH KERNEL TBD

    "gabor, sobel, schmidit, gaussian, *five_one"

    """ORDERING: FLEXILITY, FORWARD FUNCTION, TAKES IN LIST, THEN DEFINE WHATS 
    IN THE LIST, COMBINATION WITHIN LIST"""

    """Instaniating filters later on, not setting filters"""

    """WE MIGHT NEED TO FIGURE IF WE NEED TO PASS it in"""
    def forward(self, x):
        pass


class middleBlock:

    "kernel1, kernel2, kernel3, kernel4, kernel5"


    def forward():
        pass