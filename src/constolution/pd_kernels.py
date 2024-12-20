import numpy as np
from typing import Optional, Union, Tuple
from enum import Enum, auto
import torch
import cv2

class Kernels(Enum):
    Gaussian = auto()
    VerticalEdge = auto()
    HorizontalEdge = auto()
    SobelVerticalEdge = auto()
    SobelHorizontalEdge = auto()
    Box = auto()
    Gabor = auto()
    Identity = auto()
    Schmid = auto()
    Laplacian = auto()
    Average = auto()
    random_basis_gaussian = auto()
    random_basis_uniform = auto()
    random_basis_gaussian_sparse = auto()
    random_basis_uniform_sparse = auto()

# TODO for the edge ones we can swap between left and right and top and bottom

def to_tensor(type: Kernels, in_channels: int, out_channels: int, spatial_size:
              Union[int, Tuple[int,int]], groups: int, **kwargs) -> torch.Tensor:
    if isinstance(spatial_size, int):
        height, width = spatial_size, spatial_size
    else:
        height, width = spatial_size[0], spatial_size[1]
    match type:
        case Kernels.Gaussian:
            kernel = gaussian(height, width, **kwargs)
        case Kernels.VerticalEdge:
            kernel = vertical_edge(height, width, **kwargs)
        case Kernels.VerticalEdge:
            kernel = horizontal_edge(height, width, **kwargs)
        case Kernels.SobelVerticalEdge:
            kernel = sobel_vertical_edge(height, width, **kwargs)
        case Kernels.SobelHorizontalEdge:
            kernel = sobel_horizontal_edge(height, width, **kwargs)
        case Kernels.Box:
            kernel = box(height, width, **kwargs)
        case Kernels.Gabor:
            kernel = gabor(height, width, **kwargs)
        case Kernels.Identity:
            kernel = identity(height, width, **kwargs)
        case Kernels.Average:
            kernel = average(height, width, **kwargs)
        case Kernels.Schmid:
            kernel = schmid(height, width, **kwargs)
        case Kernels.Laplacian:
            kernel = laplacian(height, width, **kwargs)
        case Kernels.random_basis_gaussian:
            kernel = random_basis_gaussian(height, width, **kwargs)
        case Kernels.random_basis_uniform:
            kernel = random_basis_uniform(height, width, **kwargs)
        case Kernels.random_basis_gaussian_sparse:
            kernel = random_basis_gaussian_sparse(height, width, **kwargs)
        case Kernels.random_basis_uniform_sparse:
            kernel = random_basis_uniform_sparse(height, width, **kwargs)
        case _:
            raise ValueError(f'unsupported kernel type: {type}')

    kernel = torch.tile(torch.Tensor(kernel), (out_channels, in_channels //
                                               groups, 1, 1))
    return kernel


def gaussian(height, width, sigma=1.0) -> np.ndarray:
    y, x = np.meshgrid(
        np.linspace(-(height // 2), height // 2, height),
        np.linspace(-(width // 2), width // 2, width),
        indexing='ij'
    )

    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)

    return kernel

def vertical_edge(height, width) -> np.ndarray:
    kernel = np.zeros((height, width))

    half_width = width // 2
    kernel[:, :half_width - (1 - width % 2)] = -1
    kernel[:, half_width + 1:] = 1

    return kernel

def average(height, width, denominator: Optional[int] = None) -> np.ndarray:
    kernel = np.ones((height, width), dtype=np.float32)
    denominator = denominator or kernel.size
    kernel /= denominator
    return kernel

def sobel_horizontal_edge(height, width) -> np.ndarray:
    kernel = np.zeros((height, width))
    center_h, center_w = height // 2, width // 2
    y, x = np.meshgrid(range(height), range(width), indexing='ij')

    offset_h = 0 if height % 2 else 0.5
    offset_w = 0 if width % 2 else 0.5

    dy = y - (center_h - offset_h)
    dx = x - (center_w - offset_w)

    mask = (dy != 0) | (dx != 0)
    kernel[mask] = dy[mask] / (dy[mask]**2 + dx[mask]**2)

    if height % 2 == 0:
        kernel[center_h-1:center_h+1, :] = 0
    else:
        kernel[center_h:center_h+1, :] = 0

    return kernel

def sobel_vertical_edge(height, width) -> np.ndarray:
    kernel = np.zeros((height, width))
    center_h, center_w = height // 2, width // 2
    y, x = np.meshgrid(range(height), range(width), indexing='ij')

    offset_h = 0 if height % 2 else 0.5
    offset_w = 0 if width % 2 else 0.5

    dy = y - (center_h - offset_h)
    dx = x - (center_w - offset_w)

    mask = (dy != 0) | (dx != 0)
    kernel[mask] = dx[mask] / (dy[mask]**2 + dx[mask]**2)

    if height % 2 == 0:
        kernel[:, center_h-1:center_h+1] = 0
    else:
        kernel[:, center_h:center_h+1] = 0

    return kernel

def horizontal_edge(height, width) -> np.ndarray:
    h_k = np.zeros((height, width))
    half_width = width // 2
    h_k[:(half_width) - (1 - width % 2), :] = 1
    h_k[half_width + 1:, :] = -1

    return h_k

def box(height, width):
    return np.ones((height, width))

def gabor(height, width, sigma =1 , theta = 0, lambda_ = 1, gamma = 1,psi = 1,ktype = cv2.CV_32F)-> np.ndarray:
    kernel = cv2.getGaborKernel((width, height), sigma, theta, lambda_, gamma, psi, ktype)
    return kernel

def identity(height, width) -> np.ndarray:
    assert height == width

    kernel = np.eye(height)
    return kernel

def laplacian(height, width, sigma=1.0) -> np.ndarray:
    assert height % 2 and width % 2
    y, x = np.meshgrid(
            np.linspace(-(height // 2), height // 2, height),
            np.linspace(-(width // 2), width // 2, width),
            indexing='ij'
            )

    first_term = -1/(np.pi * sigma**4)
    second_term = 1 - ((x**2 + y**2)/(2*sigma**2))
    third_term = np.exp(-(x**2 + y**2)/(2*sigma**2))

    kernel = first_term*second_term*third_term
    return kernel/np.sum(np.abs(kernel))


def schmid(height, width, sigma=1.0, tau=1) -> np.ndarray:
    y, x = np.meshgrid(
            np.linspace(-(height // 2), height // 2, height),
            np.linspace(-(width // 2), width // 2, width),
            indexing='ij'
            )

    r = x**2 + y**2
    kernel = np.exp(-r/(2 * sigma**2)) * np.cos((2 * np.pi * tau * r) / sigma)
    return kernel/np.sum(np.abs(kernel))

def random_basis_gaussian(height, width):
    return np.random.normal(loc = 0, scale=1,size =(height,width))

def random_basis_uniform(height, width):
    return np.random.randn(height, width)

def random_basis_gaussian_sparse(height, width, threshold_pct):
    kernel = np.random.normal(loc = 0, scale=1,size =(height,width))
    kernel_pct = np.percentile(kernel, threshold_pct)
    kernel = np.where(kernel > kernel_pct, kernel, 0)
    return kernel

def random_basis_uniform_sparse(height, width, threshold_pct):
    kernel = np.random.randn(height, width)
    kernel_pct = np.percentile(kernel, threshold_pct)
    kernel = np.where(kernel > kernel_pct, kernel, 0)
    return kernel
