#!/usr/bin/env python3
"""
Module for randomly adjusting the contrast of an image
using TensorFlow.
"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjust the contrast of an image.

    Args:
        image (tf.Tensor): A 3D tensor representing an image
                           (height, width, channels).
        lower (float): Lower bound for the random contrast factor.
        upper (float): Upper bound for the random contrast factor.

    Returns:
        tf.Tensor: The contrast-adjusted image.
    """
    return tf.image.random_contrast(image, lower, upper)
