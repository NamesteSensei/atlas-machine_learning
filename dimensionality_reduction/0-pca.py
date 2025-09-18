#!/usr/bin/env python3
"""PCA that preserves a given fraction of variance"""
import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on a dataset to maintain a given fraction of variance.

    Parameters:
    - X: np.ndarray of shape (n, d), centered data
    - var: fraction of variance to preserve (float between 0 and 1)

    Returns:
    - W: np.ndarray of shape (d, nd) for projecting to lower-dimensional space
    """
    # Step 1: Compute covariance matrix
    cov = np.cov(X, rowvar=False)

    # Step 2: Eigendecomposition
    eig_vals, eig_vecs = np.linalg.eigh(cov)  # Use eigh since cov is symmetric

    # Step 3: Sort eigenvalues and eigenvectors in descending order
    sorted_idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_idx]
    eig_vecs = eig_vecs[:, sorted_idx]

    # Step 4: Compute cumulative variance ratio
    cum_var = np.cumsum(eig_vals) / np.sum(eig_vals)

    # Step 5: Find the number of components to retain desired variance
    num_components = np.searchsorted(cum_var, var) + 1

    # Step 6: Return projection matrix W
    W = eig_vecs[:, :num_components]
    return W
