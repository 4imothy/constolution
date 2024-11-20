from typing import Tuple, Union
from torch import nn
from . import _utils

Kernels = _utils.Kernels

class Constolution(nn.Module):

    # TODO see what hparams make sense for const padding, zeros, padding_mode, bias
    def __init__(self, type: Kernels, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]]):
        print('we are live')
        self.weight = _utils.to_tensor(type, in_channels, out_channels)

