#!/usr/bin/env python3
"""Performs a convolution on grayscale images with stride and padding"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with stride and padding.

    Parameters:
    - images: numpy.ndarray of shape (m, h, w)
    - kernel: numpy.ndarray of shape (kh, kw)
    - padding: either 'same', 'valid', or a tuple of (ph, pw)
    - stride: tuple of (sh, sw)

    Returns:
    - numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Handle padding
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        raise ValueError("padding must be 'same', 'valid', or a (ph, pw) tuple")

    # Pad images
    padded_images = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Compute output shape
    out_h = ((h + 2 * ph - kh) // sh) + 1
    out_w = ((w + 2 * pw - kw) // sw) + 1
    output = np.zeros((m, out_h, out_w))

    # Convolution loops
    for i in range(out_h):
        for j in range(out_w):
            img_slice = padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            output[:, i, j] = np.sum(img_slice * kernel, axis=(1, 2))

    return output
