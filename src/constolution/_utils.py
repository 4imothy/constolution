import torch
from enum import Enum

class ConstKernel(Enum):
    Gaussian: int

# TODO should the kernels have different input/out channels should it always be depthwise?
def kernel_to_tensor(type: ConstKernel, in_channels: int, out_channels: int) -> torch.Tensor:
    print('we are live')
