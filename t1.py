import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

image = np.array([[3, 4, 8, 15, 25, 44, 50, 52] for _ in range(8)])


def convolute(image, kernel):
    g_image = []
    kernel_size = len(kernel)
    pad = kernel_size // 2  # 3 // 2 = 1
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    # print(image)
    # print(image.shape)
    # for i in range(pad, image.shape[0] - pad):
    #     row = []
    #     sum = 0
    #     for j in range(pad, image.shape[1] - pad):
    #         for ky in range(- 1 * pad, pad + 1):
    #             for kx in range(- 1 * pad, pad + 1):
    #                 # print(i+ky, j+kx, pad-ky, pad-kx)
    #                 # print(image[i + ky][j + kx], "*", kernel[pad - ky][pad - kx])
    #                 sum += int (image[i + ky][j + kx] * kernel[pad - ky][pad - kx])
    #         row.append(sum)
    #         # print(sum)
    #         # print('-' * 50)
    #     g_image.append(row)
    g_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            sum = 0
            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    ny, nx = i + ky - pad, j + kx - pad
                    sum += padded_image[ny + pad][nx + pad] * kernel[ky][kx]
                    g_image[i][j] = sum
    return np.array(g_image)


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
# gradient_y = convolute(image, prewitt_hy)

gradient_x_sci = convolve2d(image, prewitt_x, boundary='fill', fillvalue=0)

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 2]
])
sobel_y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

plt.imshow(image, interpolation='nearest')
# plt.show
# plt.imshow(convolve2d(image, sobel_x, boundary='fill', fillvalue=0), interpolation='nearest')
# plt.show()

print(convolute(image, sobel_x))
print(ndimage.convolve(image, sobel_x))
