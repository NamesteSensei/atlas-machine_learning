#!/usr/bin/env python3
"""
Performs Principal Component Analysis (PCA)
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Parameters:
    - X: np.ndarray of shape (n, d), centered data matrix
    - var: float, fraction of variance to preserve (0 < var <= 1)

    Returns:
    - W: np.ndarray of shape (d, nd), weight matrix for projecting
    """
    # Compute the covariance matrix
    cov = np.cov(X, rowvar=False)

    # Eigendecomposition of the covariance matrix
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_idx]
    eig_vecs = eig_vecs[:, sorted_idx]

    # Compute the cumulative variance ratio
    cum_var = np.cumsum(eig_vals) / np.sum(eig_vals)

    # Determine the number of components to retain
    num_components = np.searchsorted(cum_var, var) + 1

    # Select the top eigenvectors
    W = eig_vecs[:, :num_components]

    return W
