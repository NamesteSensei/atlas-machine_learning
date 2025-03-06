#!/usr/bin/env python3

"""
Module: 3-one_hot
Converts a numeric label vector into a one-hot matrix.
"""

import tensorflow.keras as K


def one_hot(labels, classes):
    """
    Converts a label vector into a one-hot matrix.

    Parameters:
    - labels (numpy.ndarray): Array of shape (m,) containing the numeric
      class labels. 'm' is the number of examples.
    - classes (int): The total number of classes.

    Returns:
    - numpy.ndarray: One-hot matrix of shape (m, classes) or None on failure.
    """
    try:
        return K.utils.to_categorical(labels, num_classes=classes)
    except Exception:
        return None
