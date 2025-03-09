#!/usr/bin/env python3
"""
Performs a same convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
        images (np.ndarray): (m, h, w) grayscale images.
        kernel (np.ndarray): (kh, kw) convolution kernel.

    Returns:
        np.ndarray: The convolved images.
    """
    num_images, img_height, img_width = images.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate padding for same convolution
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad images with zeros
    padded_images = np.pad(
        images,
        pad_width=((0, 0), (pad_height, pad_height), (pad_width, pad_width)),
        mode='constant'
    )

    # Initialize the output array
    convolved_images = np.zeros((num_images, img_height, img_width))

    # Perform convolution using two loops
    for row in range(img_height):
        for col in range(img_width):
            convolved_images[:, row, col] = np.sum(
                padded_images[:, row:row+kernel_height, col:col+kernel_width]
                * kernel, axis=(1, 2)
            )

    return convolved_images
