import math

import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

image = np.array([[3, 4, 8, 15, 25, 44, 50, 52] for _ in range(8)])
image2 = np.array([
    [7, 12, 9],
    [6, 7, 8],
    [3, 4, 5]
])


def convolute(image, kernel):
    kernel_size = len(kernel)
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    g_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            total = 0
            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    ny, nx = i + ky - pad, j + kx - pad
                    total += padded_image[ny + pad][nx + pad] * kernel[ky][kx]
            g_image[i][j] = total
    return np.array(g_image)

def q1q2():
    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    prewitt_y = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])

    gradient_x = convolute(image, prewitt_x)
    gradient_y = convolute(image, prewitt_y)

    print("Prewitt: ")
    print("Gradient Magnitude")
    print(np.sqrt(gradient_x ** 2 + gradient_y ** 2)[1:-1, 1:-1])
    gradient_x = convolute(image2, prewitt_x)
    gradient_y = convolute(image2, prewitt_y)
    print("Gradient Direction")
    print(math.degrees(np.arctan2(gradient_y[1:-1, 1:-1], gradient_x[1:-1, 1:-1])))

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    gradient_x = convolute(image, sobel_x)
    gradient_y = convolute(image, sobel_y)

    print("Sobel: ")
    print("Gradient Magnitude")
    print(np.sqrt(gradient_x ** 2 + gradient_y ** 2)[1:-1, 1:-1])
    gradient_x = convolute(image2, sobel_x)
    gradient_y = convolute(image2, sobel_y)
    print("Gradient Direction")
    print(math.degrees(np.arctan2(gradient_y[1:-1, 1:-1], gradient_x[1:-1, 1:-1])))

    laplacian = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    print("Laplacian: ", convolute(image, laplacian)[1:-1, 1:-1])

def q3():
    gaussian_mask = np.array([
        [1/36, 1/9, 1/36],
        [1/9, 4/9, 1/9],
        [1/36, 1/9, 1/36]])
    print("Gaussian: ", convolute(image, gaussian_mask)[1:-1, 1:-1])

q3()