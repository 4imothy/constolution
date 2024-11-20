from typing import Union, Tuple
import torch
from enum import Enum, auto

class Kernels(Enum):
    Gaussian = auto()

# TODO should the kernels have different input/out channels should it always be depthwise?
def to_tensor(type: Kernels, in_channels: int, out_channels: int, spatial_size: Union[int, Tuple[int,int]]) -> torch.Tensor:
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    kernel = torch.zeros((out_channels, in_channels, *spatial_size))
    match type:
        case Kernels.Gaussian:
            sigma = 1.0
            height, width = spatial_size
            center_x, center_y = (width - 1) / 2, (height - 1) / 2

            y = torch.arange(height, dtype=torch.float32)
            x = torch.arange(width, dtype=torch.float32)
            yy, xx = torch.meshgrid(y, x, indexing="ij")

            kernel = torch.exp(
                -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2)
            )

            kernel /= kernel.sum()

            kernel = kernel.unsqueeze(0).unsqueeze(0)
            kernel = torch.tile(kernel, (out_channels, in_channels, 1, 1))
        case _:
            raise ValueError(f'unsupported kernel type: {type}')


    return kernel
