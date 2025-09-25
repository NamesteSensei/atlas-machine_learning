#!/usr/bin/env python3
"""Performs pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Parameters:
    - images: numpy.ndarray of shape (m, h, w, c)
    - kernel_shape: tuple (kh, kw)
    - stride: tuple (sh, sw)
    - mode: 'max' or 'avg'

    Returns:
    - numpy.ndarray of shape (m, out_h, out_w, c) with pooled results
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    pooled = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            img_slice = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                pooled[:, i, j, :] = np.max(img_slice, axis=(1, 2))
            else:  # avg
                pooled[:, i, j, :] = np.mean(img_slice, axis=(1, 2))

    return pooled
