#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_channels = __import__(
    '4-convolve_channels'
).convolve_channels

if __name__ == '__main__':
    dataset = np.load('animals_1.npz')
    images = dataset['data']
    print(images.shape)  # Expected: (10000, 32, 32, 3)

    kernel = np.array([
        [[0, 0, 0], [-1, -1, -1], [0, 0, 0]],
        [[-1, -1, -1], [5, 5, 5], [-1, -1, -1]],
        [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]
    ])

    images_conv = convolve_channels(
        images, kernel, padding='valid', stride=(1, 1)
    )
    print(images_conv.shape)  # Expected: (10000, 30, 30)

    # Display original and convolved images
    plt.imshow(images[0])
    plt.title('Original Image')
    plt.show()

    plt.imshow(images_conv[0], cmap='gray')
    plt.title('Convolved Image')
    plt.show()
