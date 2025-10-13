#!/usr/bin/env python3
"""Gaussian Process - Initialization"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):  # noqa: E741
        """Initialize Gaussian Process

        X_init: np.ndarray of shape (t, 1) - initial inputs
        Y_init: np.ndarray of shape (t, 1) - initial outputs
        l: float - kernel length parameter
        sigma_f: float - output standard deviation
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l  # noqa: E741
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculate covariance matrix using RBF kernel"""
        sqdist = (np.sum(X1 ** 2, 1).reshape(-1, 1)
                  + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T))
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)
