#!/usr/bin/env python3

"""
Module: 3-one_hot
Converts a label vector into a one-hot matrix.
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Parameters:
    - labels (numpy.ndarray): Array containing the numeric class labels.
    - classes (int, optional): The total number of classes. If not provided,
      it will be inferred from the labels.

    Returns:
    - numpy.ndarray: One-hot matrix with the last dimension as the number
      of classes.
    """
    try:
        return K.utils.to_categorical(labels, num_classes=classes)
    except Exception:
        return None
