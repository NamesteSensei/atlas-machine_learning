#!/usr/bin/env python3
import numpy as np

def pca(X, ndim):
    """
    Performs PCA on a dataset to reduce to a specified number of dimensions.

    Parameters:
    - X: np.ndarray of shape (n, d)
    - ndim: int, number of dimensions to keep

    Returns:
    - T: np.ndarray of shape (n, ndim), transformed data
    """
    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    W = Vt[:ndim].T
    return np.matmul(X_centered, W)
