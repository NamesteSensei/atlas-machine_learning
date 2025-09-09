#!/usr/bin/env python3
"""
MultiNormal: multivariate normal distribution with PDF evaluation.
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

    def pdf(self, x):
        """
        Calculate the PDF at a given data point.

        Parameters
        ----------
        x : numpy.ndarray
            Array of shape (d, 1) representing a data point.

        Returns
        -------
        float
            The value of the PDF at point x.

        Raises
        ------
        TypeError
            If x is not a numpy.ndarray.
        ValueError
            If x does not have shape (d, 1).
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        # Compute PDF using multivariate Gaussian formula
        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)

        norm_const = 1.0 / np.sqrt(((2 * np.pi) ** d) * det_cov)
        diff = x - self.mean
        exponent = -0.5 * (diff.T @ inv_cov @ diff)

        return float(norm_const * np.exp(exponent))
