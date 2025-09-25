#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid

if __name__ == '__main__':
    dataset = np.load('MNIST.npz')  # Ensure MNIST.npz is in the same dir
    images = dataset['X_train']     # Shape: (50000, 28, 28)

    print(images.shape)  # Expect: (50000, 28, 28)

    # Kernel to detect vertical edges (Sobel-like)
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    images_conv = convolve_grayscale_valid(images, kernel)

    print(images_conv.shape)  # Expect: (50000, 26, 26)

    # Visualize original and convolved image
    plt.imshow(images[0], cmap='gray')
    plt.title("Original Image")
    plt.show()

    plt.imshow(images_conv[0], cmap='gray')
    plt.title("Convolved Image (Valid)")
    plt.show()
