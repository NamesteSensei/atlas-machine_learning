#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Importing the function using the required import method
convolve_grayscale_valid = __import__(
    '0-convolve_grayscale_valid'
).convolve_grayscale_valid

if __name__ == '__main__':
    # Load the MNIST dataset
    dataset = np.load('MNIST.npz')
    images = dataset['X_train']
    print(images.shape)  # Expected output: (50000, 28, 28)

    # Define the kernel for edge detection
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    # Perform the valid convolution
    images_conv = convolve_grayscale_valid(images, kernel)
    print(images_conv.shape)  # Expected output: (50000, 26, 26)

    # Display the original and convolved images
    plt.imshow(images[0], cmap='gray')
    plt.show()
    
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
