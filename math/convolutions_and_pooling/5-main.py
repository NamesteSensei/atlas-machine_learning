#!/usr/bin/env python3
import numpy as np
convolve = __import__('5-convolve').convolve

if __name__ == '__main__':
    dataset = np.load('animals_1.npz')
    images = dataset['data']
    print(images.shape)  # (10000, 32, 32, 3)

    # 3 kernels for testing
    kernels = np.array([
        [[[0, 1, 1], [0, 1, 1], [0, 1, 1]],
         [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
         [[0, -1, 1], [0, -1, 1], [0, -1, 1]]],

        [[[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]],
         [[5, 0, 0], [5, 0, 0], [5, 0, 0]],
         [[-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]],

        [[[0, 1, -1], [0, 1, -1], [0, 1, -1]],
         [[-1, 0, -1], [-1, 0, -1], [-1, 0, -1]],
         [[0, -1, -1], [0, -1, -1], [0, -1, -1]]]
    ])

    images_conv = convolve(images, kernels, padding='valid')
    print(images_conv.shape)  # Expect: (10000, 30, 30, 3)
    print(images_conv[0, :, :, 0])  # first kernel output
