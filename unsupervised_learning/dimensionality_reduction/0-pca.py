#!/usr/bin/env python3
"""
Performs PCA on a dataset to retain a given fraction of total variance.
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset to maintain a given variance fraction.

    Parameters:
    - X: np.ndarray of shape (n, d), zero-mean data
    - var: float, the fraction of variance to retain

    Returns:
    - W: np.ndarray of shape (d, nd), the weights matrix for projection
    """
    # Perform SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute variance explained
    variance_explained = (S ** 2) / np.sum(S ** 2)
    cumulative_variance = np.cumsum(variance_explained)

    # Find number of components to retain var amount of variance
    nd = np.searchsorted(cumulative_variance, var) + 1

    # Get the top nd principal components (columns)
    W = Vt[:nd].T  # shape (d, nd)

    return W
