#!/usr/bin/env python3
import numpy as np

"""
/**
   This module provides a function to convert a numeric label vector into a one-hot matrix.
*/
"""

def one_hot_encode(Y, classes):
    """
    /**
       Converts a numeric label vector into a one-hot matrix.
       Y is a numpy.ndarray with shape (m,) containing numeric labels.
       classes indicates the total count of classes.
       Returns a one-hot encoded matrix with shape (classes, m), or None on failure.
    */
    """
    if type(Y) is not np.ndarray or Y.ndim != 1:
        return None
    m = Y.shape[0]
    try:
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None
