#!/usr/bin/env python3
"""
Defines the MultiNormal class representing a multivariate normal distribution.
"""

import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal (Gaussian) distribution.
    """

    def __init__(self, data):
        """
        Initializes a MultiNormal instance.

        Parameters:
        - data (numpy.ndarray): A 2D array of shape (d, n) containing the dataset,
          where d is the number of dimensions and n is the number of data points.

        Sets:
        - self.mean: numpy.ndarray of shape (d, 1) containing the mean vector.
        - self.cov: numpy.ndarray of shape (d, d) containing the covariance matrix.

        Raises:
        - TypeError: if data is not a 2D numpy.ndarray
        - ValueError: if n (number of data points) is less than 2
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate the mean vector (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Center the data and calculate the covariance matrix (d, d)
        data_centered = data - self.mean
        self.cov = (data_centered @ data_centered.T) / (n - 1)
