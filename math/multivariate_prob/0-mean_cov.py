#!/usr/bin/env python3
"""
Module that provides a function to compute the mean vector and covariance
matrix of a multivariate dataset without using numpy.cov.
"""

import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Parameters:
    - X (numpy.ndarray): A 2D array of shape (n, d) where:
        - n is the number of data points
        - d is the number of dimensions per data point

    Returns:
    - mean (numpy.ndarray): A 1-row array of shape (1, d) representing
      the mean vector of the dataset.
    - cov (numpy.ndarray): A square array of shape (d, d) representing
      the covariance matrix of the dataset.

    Raises:
    - TypeError: If X is not a 2D numpy.ndarray
    - ValueError: If the number of data points (n) is less than 2
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)  # shape (1, d)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / (n - 1)  # shape (d, d)

    return mean, cov
