#!/usr/bin/env python3
"""
Performs PCA on a dataset to retain a given fraction of total variance.
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Parameters:
    - X: np.ndarray of shape (n, d), dataset with mean already removed
    - var: float, the fraction of variance to preserve (default 0.95)

    Returns:
    - W: np.ndarray of shape (d, nd), the weights matrix that maintains
         `var` fraction of X's variance
    """
    # Compute SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Variance explained by each singular value
    explained_variance = (S ** 2) / np.sum(S ** 2)
    cumulative_variance = np.cumsum(explained_variance)

    # Minimum number of components that reach the target variance
    nd = np.where(cumulative_variance >= var)[0][0] + 1

    # Weight matrix (principal components as columns)
    W = Vt[:nd].T

    return W
