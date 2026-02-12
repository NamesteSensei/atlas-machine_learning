#!/usr/bin/env python3
"""
Module for performing a random crop on an image using TensorFlow.
"""

import tensorflow as tf


def crop_image(image, size):
    """
    Perform a random crop on an image.

    Args:
        image (tf.Tensor): A 3D tensor representing an image
                           (height, width, channels).
        size (tuple): A tuple representing the desired crop size
                      (crop_height, crop_width, channels).

    Returns:
        tf.Tensor: The randomly cropped image.
    """
    return tf.image.random_crop(image, size)
