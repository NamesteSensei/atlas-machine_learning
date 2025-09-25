#!/usr/bin/env python3
"""Performs a convolution on images using multiple kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Parameters:
    - images: numpy.ndarray of shape (m, h, w, c)
    - kernels: numpy.ndarray of shape (kh, kw, c, nc)
    - padding: 'same', 'valid', or (ph, pw)
    - stride: tuple (sh, sw)

    Returns:
    - numpy.ndarray containing the convolved images
      with shape (m, out_h, out_w, nc)
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("Kernel depth must match image channels")

    # Handle padding
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        raise ValueError("padding must be 'same', 'valid', or (ph, pw)")

    # Pad images
    padded_images = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Output shape
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, out_h, out_w, nc))

    # Convolution
    for i in range(out_h):
        for j in range(out_w):
            img_slice = padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            for k in range(nc):  # one loop for each kernel
                output[:, i, j, k] = np.sum(
                    img_slice * kernels[..., k],
                    axis=(1, 2, 3)
                )

    return output
