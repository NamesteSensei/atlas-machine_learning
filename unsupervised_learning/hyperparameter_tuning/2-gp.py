#!/usr/bin/env python3
"""
Gaussian Process with update method.
"""

import numpy as np


class GaussianProcess:
    """Implements a 1-D Gaussian Process with updating."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize GP."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """RBF kernel."""
        sqdist = (np.sum(X1 ** 2, 1).reshape(-1, 1)
                  + np.sum(X2 ** 2, 1)
                  - 2 * np.dot(X1, X2.T))
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """Predict mean and variance."""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)
        mu = K_s.T @ K_inv @ self.Y
        cov = K_ss - K_s.T @ K_inv @ K_s
        return mu.ravel(), np.diag(cov)

    def update(self, X_new, Y_new):
        """
        Add a new data point and update the kernel matrix.
        """
        self.X = np.vstack((self.X, X_new.reshape(-1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(-1, 1)))
        self.K = self.kernel(self.X, self.X)
