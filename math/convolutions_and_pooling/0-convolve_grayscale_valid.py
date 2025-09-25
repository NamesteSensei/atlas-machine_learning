#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Convolve images with a kernel using 'valid' padding (no padding)."""
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Output dimensions
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize output
    output = np.zeros((m, output_h, output_w))

    # Loop over image height and width
    for i in range(output_h):
        for j in range(output_w):
            # Slice all images at once using broadcasting
            img_slice = images[:, i:i+kh, j:j+kw]  # shape: (m, kh, kw)
            output[:, i, j] = np.sum(img_slice * kernel, axis=(1, 2))

    return output
