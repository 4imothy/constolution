import numpy as np

def gaussian_kernel(height, width, sigma=1.0):
    y, x = np.meshgrid(
        np.linspace(-(height // 2), height // 2, height),
        np.linspace(-(width // 2), width // 2, width),
        indexing="ij"
    )

    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)


    return kernel

def horizontal_kernel(height, width):
    pass 

def box_kernel(height, width):
    pass

def vertical_kernel(height, width):
    pass

def sobel_kernel(height, width):
    pass

def gabor_kernel(height, width):
    pass 

