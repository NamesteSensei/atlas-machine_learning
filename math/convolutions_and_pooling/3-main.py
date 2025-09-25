#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale

if __name__ == '__main__':
    dataset = np.load('MNIST.npz')
    images = dataset['X_train']
    print(images.shape)  # (50000, 28, 28)

    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    images_conv = convolve_grayscale(images, kernel, padding='valid', stride=(2, 2))
    print(images_conv.shape)  # Expect: (50000, 13, 13)

    # Save images
    plt.imsave("task3_original.png", images[0], cmap='gray')
    plt.imsave("task3_stride_conv.png", images_conv[0], cmap='gray')
