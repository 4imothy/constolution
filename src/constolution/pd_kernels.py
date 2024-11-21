import numpy as np

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

def sobel_horizontal(height, width) -> np.ndarray:
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

def sobel_vertical(height, width) -> np.ndarray:
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

def horizontal_kernel(height, width) -> np.ndarray:
    h_k = np.zeros((height, width))
    half_width = width // 2
    h_k[:(half_width) - (1 - width % 2), :] = 1
    h_k[half_width + 1:, :] = -1

    return h_k

def box_kernel(height, width):
    return np.ones((height, width)) / (height * width)

def gabor_kernel(height, width):
    pass

