#!/usr/bin/env python3
"""
Module for shuffling data points in two matrices the same way.

This ensures that inputs (X) and corresponding labels (Y)
remain aligned after shuffling.
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles data points in two matrices in the same way.

    Parameters:
    X (numpy.ndarray): Matrix of shape (m, nx) where:
        - m is the number of data points
        - nx is the number of features in X
    Y (numpy.ndarray): Matrix of shape (m, ny) where:
        - m is the number of data points
        - ny is the number of features in Y

    Returns:
    tuple: (shuffled_X, shuffled_Y) with data shuffled in the same way
    """
    perm = np.random.permutation(X.shape[0])  # Generate shuffled indices
    return X[perm], Y[perm]
