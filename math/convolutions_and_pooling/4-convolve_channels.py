#!/usr/bin/env python3
"""
Performs a convolution on images with multiple channels.
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with multiple channels.

    Args:
        images (np.ndarray): (m, h, w, c) images with channels.
        kernel (np.ndarray): (kh, kw, c) convolution kernel.
        padding (str or tuple): Padding ('same', 'valid') or (ph, pw).
        stride (tuple): (sh, sw) stride for height and width.

    Returns:
        np.ndarray: The convolved images.
    """
    m, img_h, img_w, img_c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    # Ensure the kernel's depth matches the image's channels
    assert img_c == kc, "Kernel depth must match image channels"

    # Determine padding
    if padding == 'same':
        ph = ((img_h - 1) * sh + kh - img_h) // 2
        pw = ((img_w - 1) * sw + kw - img_w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad the images
    padded_imgs = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Calculate output dimensions
    out_h = (img_h + 2 * ph - kh) // sh + 1
    out_w = (img_w + 2 * pw - kw) // sw + 1

    # Initialize the output array
    convolved_imgs = np.zeros((m, out_h, out_w))

    # Perform convolution using two loops
    for i in range(out_h):
        for j in range(out_w):
            convolved_imgs[:, i, j] = np.sum(
                padded_imgs[
                    :, i * sh:i * sh + kh, j * sw:j * sw + kw, :
                ] * kernel, axis=(1, 2, 3)
            )

    return convolved_imgs
