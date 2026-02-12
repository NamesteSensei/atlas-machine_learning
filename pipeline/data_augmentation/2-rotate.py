#!/usr/bin/env python3
"""
Module for rotating an image 90 degrees counter-clockwise
using TensorFlow.
"""

import tensorflow as tf


def rotate_image(image):
    """
    Rotate an image 90 degrees counter-clockwise.

    Args:
        image (tf.Tensor): A 3D tensor representing an image
                           (height, width, channels).

    Returns:
        tf.Tensor: The rotated image.
    """
    return tf.image.rot90(image, k=1)
