#!/usr/bin/env python3

"""
Performs a valid convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images (np.ndarray): (m, h, w) containing multiple grayscale images.
        kernel (np.ndarray): (kh, kw) containing the convolution kernel.

    Returns:
        np.ndarray: The convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize the output array
    convolved = np.zeros((m, output_h, output_w))

    # Perform convolution using only two loops
    for i in range(output_h):
        for j in range(output_w):
            convolved[:, i, j] = np.sum(
                images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )

    return convolved
