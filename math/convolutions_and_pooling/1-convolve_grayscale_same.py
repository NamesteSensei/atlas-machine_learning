#!/usr/bin/env python3
"""Performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Parameters:
    - images: numpy.ndarray of shape (m, h, w)
    - kernel: numpy.ndarray of shape (kh, kw)

    Returns:
    - numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding
    ph = kh // 2
    pw = kw // 2

    # Pad images with zeros
    padded_images = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Output dimensions remain same as input
    output = np.zeros((m, h, w))

    # Loop through positions
    for i in range(h):
        for j in range(w):
            img_slice = padded_images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(img_slice * kernel, axis=(1, 2))

    return output
