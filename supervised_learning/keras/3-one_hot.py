#!/usr/bin/env python3
"""
3-one_hot.py
------------
Converts a label vector into a one-hot encoded matrix.
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Parameters
    ----------
    labels : array-like
        Vector of integer class labels.
    classes : int, optional
        Total number of classes. If None, inferred from labels.

    Returns
    -------
    one_hot_matrix : ndarray
        One-hot encoded matrix with shape (len(labels), classes).
    """
    return K.utils.to_categorical(labels, num_classes=classes)
