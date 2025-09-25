#!/usr/bin/env python3
"""
This module performs Principal Component Analysis (PCA)
for reducing the dimensionality of datasets using SVD.
"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset to reduce it to a given number of dimensions.

    Parameters:
    - X: np.ndarray of shape (n, d), the dataset
    - ndim: int, the target number of dimensions

    Returns:
    - T: np.ndarray of shape (n, ndim), the transformed dataset
    """
    # Center the data
    X_mean = X - np.mean(X, axis=0)

    # Perform SVD
    U, S, Vt = np.linalg.svd(X_mean, full_matrices=False)

    # Project the data onto the top ndim principal components
    W = Vt[:ndim].T
    T = np.matmul(X_mean, W)

    return T
