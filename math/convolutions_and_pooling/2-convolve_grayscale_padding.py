#!/usr/bin/env python3
"""
Performs a convolution on grayscale images with custom padding.
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images (np.ndarray): (m, h, w) grayscale images.
        kernel (np.ndarray): (kh, kw) convolution kernel.
        padding (tuple): (ph, pw) padding for height and width.

    Returns:
        np.ndarray: The convolved images.
    """
    num_images, img_height, img_width = images.shape
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = padding

    # Pad images with zeros
    padded_images = np.pad(
        images,
        pad_width=((0, 0), (pad_height, pad_height), (pad_width, pad_width)),
        mode='constant'
    )

    # Calculate output dimensions
    output_height = img_height + 2 * pad_height - kernel_height + 1
    output_width = img_width + 2 * pad_width - kernel_width + 1

    # Initialize the output array
    convolved_images = np.zeros((num_images, output_height, output_width))

    # Perform convolution using two loops
    for row in range(output_height):
        for col in range(output_width):
            convolved_images[:, row, col] = np.sum(
                padded_images[:, row:row+kernel_height, col:col+kernel_width]
                * kernel, axis=(1, 2)
            )

    return convolved_images
