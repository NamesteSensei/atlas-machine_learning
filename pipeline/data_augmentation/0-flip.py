#!/usr/bin/env python3
"""
Module for flipping an image horizontally using TensorFlow.
"""

import tensorflow as tf


def flip_image(image):
    """
    Flip an image horizontally.

    Args:
        image (tf.Tensor): A 3D tensor representing an image
                           (height, width, channels).

    Returns:
        tf.Tensor: The horizontally flipped image.
    """
    return tf.image.flip_left_right(image)
