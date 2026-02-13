#!/usr/bin/env python3
"""
Module for adjusting the hue of an image using TensorFlow.
"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Adjust the hue of an image.

    Args:
        image (tf.Tensor): A 3D tensor representing an image
                           (height, width, channels).
        delta (float): Amount to shift the hue. Must be in
                       the interval [-0.5, 0.5].

    Returns:
        tf.Tensor: The hue-adjusted image.
    """
    return tf.image.adjust_hue(image, delta)
