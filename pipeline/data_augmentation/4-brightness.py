#!/usr/bin/env python3
"""
Module for randomly adjusting the brightness of an image
using TensorFlow.
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly adjust the brightness of an image.

    Args:
        image (tf.Tensor): A 3D tensor representing an image
                           (height, width, channels).
        max_delta (float): Maximum brightness change.

    Returns:
        tf.Tensor: The brightness-adjusted image.
    """
    return tf.image.random_brightness(image, max_delta)
