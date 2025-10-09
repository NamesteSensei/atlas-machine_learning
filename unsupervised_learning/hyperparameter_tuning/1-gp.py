#!/usr/bin/env python3
"""
Gaussian Process with prediction capability.
"""

import numpy as np


class GaussianProcess:
    """Extends the basic GP with predictive methods."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize attributes."""
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
        """
        Predict mean and variance for new points X_s.
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T @ K_inv @ self.Y
        cov = K_ss - K_s.T @ K_inv @ K_s
        return mu.ravel(), np.diag(cov)
