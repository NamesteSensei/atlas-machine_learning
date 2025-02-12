#!/usr/bin/env python3
import numpy as np

"""
/**
   This module provides a function to convert a one-hot matrix into a label vector.
*/
"""

def one_hot_decode(one_hot):
    """
    /**
       Converts a one-hot encoded numpy.ndarray into a numeric label vector.
       one_hot is a numpy.ndarray with shape (classes, m).
       Returns a numpy.ndarray with shape (m,) containing the labels, or None on failure.
    */
    """
    if type(one_hot) is not np.ndarray or one_hot.ndim != 2:
        return None
    try:
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception:
        return None
