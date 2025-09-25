#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_padding = __import__('2-convolve_grayscale_padding').convolve_grayscale_padding

if __name__ == '__main__':
    dataset = np.load('MNIST.npz')
    images = dataset['X_train']
    print(images.shape)  # (50000, 28, 28)

    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    images_conv = convolve_grayscale_padding(images, kernel, (2, 4))
    print(images_conv.shape)  # Expect: (50000, 30, 34)

    # Save images
    plt.imsave("task2_original.png", images[0], cmap='gray')
    plt.imsave("task2_padding_conv.png", images_conv[0], cmap='gray')
