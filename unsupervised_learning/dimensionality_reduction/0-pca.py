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
    cov = np.cov(X, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    sorted_idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_idx]
    eig_vecs = eig_vecs[:, sorted_idx]

    cum_var = np.cumsum(eig_vals) / np.sum(eig_vals)
    num_components = np.argmax(cum_var >= var) + 1

    W = eig_vecs[:, :num_components]
    return W
