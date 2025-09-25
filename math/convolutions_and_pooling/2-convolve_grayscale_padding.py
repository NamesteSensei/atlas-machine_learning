#!/usr/bin/env python3
"""Performs a convolution on grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Parameters:
    - images: numpy.ndarray of shape (m, h, w)
    - kernel: numpy.ndarray of shape (kh, kw)
    - padding: tuple of (ph, pw)

    Returns:
    - numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad with zeros
    padded_images = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Output shape
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1
    output = np.zeros((m, output_h, output_w))

    # Convolution loops
    for i in range(output_h):
        for j in range(output_w):
            img_slice = padded_images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(img_slice * kernel, axis=(1, 2))

    return output
