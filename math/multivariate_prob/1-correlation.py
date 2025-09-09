#!/usr/bin/env python3
"""
Module to compute the correlation matrix from a given covariance matrix.
"""

import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix.

    Parameters:
    - C (numpy.ndarray): A 2D square array of shape (d, d) representing
      the covariance matrix.

    Returns:
    - numpy.ndarray: A 2D array of shape (d, d) representing the
      correlation matrix.

    Raises:
    - TypeError: If C is not a numpy.ndarray
    - ValueError: If C is not a 2D square matrix (shape != (d, d))
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    stddev = np.sqrt(np.diag(C))

    if np.any(stddev == 0):
        raise ValueError("Standard deviation cannot be zero")

    denom = np.outer(stddev, stddev)
    corr = C / denom

    return corr
