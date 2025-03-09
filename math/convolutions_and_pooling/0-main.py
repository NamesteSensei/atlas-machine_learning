#!/usr/bin/env python3

import numpy as np
convolve_grayscale_valid = __import__(
    '0-convolve_grayscale_valid'
).convolve_grayscale_valid

if __name__ == '__main__':
    # Load the MNIST dataset
    dataset = np.load('MNIST.npz')
    images = dataset['X_train']
    print(images.shape)  # Expected: (50000, 28, 28)

    # Define the kernel for edge detection
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    # Perform the valid convolution
    images_conv = convolve_grayscale_valid(images, kernel)
    print(images_conv.shape)  # Expected: (50000, 26, 26)

    # Print the first convolved image to match the expected output format
    print(images_conv[0])
