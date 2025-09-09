#!/usr/bin/env python3
"""
MultiNormal: multivariate normal distribution.
"""

import numpy as np


class MultiNormal:
    """
    Represents a multivariate normal (Gaussian) distribution.
    """

    def __init__(self, data):
        """
        Initialize a MultiNormal instance.

        Parameters
        ----------
        data : numpy.ndarray
            2D array of shape (d, n) with the dataset.
            d is the number of dimensions.
            n is the number of data points.

        Sets
        ----
        self.mean : numpy.ndarray
            Array of shape (d, 1) with the mean vector.
        self.cov : numpy.ndarray
            Array of shape (d, d) with the covariance matrix.

        Raises
        ------
        TypeError
            If data is not a 2D numpy.ndarray.
        ValueError
            If n (number of data points) is less than 2.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Mean vector (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Covariance matrix (d, d)
        data_centered = data - self.mean
        self.cov = (data_centered @ data_centered.T) / (n - 1)
