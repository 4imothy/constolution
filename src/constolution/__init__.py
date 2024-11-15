from typing import Tuple, Union
from torch import nn
from . import _utils

ConstKernel = _utils.ConstKernel

class Constolution(nn.Module):

    # TODO see what hparams make sense for const padding, zeros, padding_mode, bias
    def __init__(self, type: ConstKernel, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]]):
        print('we are live')

