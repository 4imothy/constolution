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

# nd array return type 
def horizontal_kernel(height, width):
    h_k = np.zeros((height, width))
    # fill rows between -2 to 2 horizontally across
    # weights on top are 1 and bottom -1
    half_width = width // 2
    h_k[:(half_width) - (1 - width % 2), :] = 1
    h_k[half_width + 1:, :] = -1 

    
    return h_k 

print(horizontal_kernel(6,6))

def box_kernel(height, width):
    pass

def vertical_kernel(height, width):
    pass

def sobel_kernel(height, width):
    pass

def gabor_kernel(height, width):
    pass 

