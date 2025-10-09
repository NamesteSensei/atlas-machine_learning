#!/usr/bin/env python3
"""
Gaussian Process Module
Implements a noiseless 1-D Gaussian process using the RBF kernel.
"""

import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1-D Gaussian Process.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initialize the Gaussian Process.

        Parameters
        ----------
        X_init : np.ndarray of shape (t, 1)
            Sample inputs.
        Y_init : np.ndarray of shape (t, 1)
            Corresponding outputs.
        l : float
            Kernel length-scale.
        sigma_f : float
            Function output standard deviation.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Compute the RBF (Radial Basis Function) kernel matrix.
        """
        sqdist = (np.sum(X1 ** 2, 1).reshape(-1, 1)
                  + np.sum(X2 ** 2, 1)
                  - 2 * np.dot(X1, X2.T))
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)
