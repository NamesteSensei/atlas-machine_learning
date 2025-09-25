#!/usr/bin/env python3
import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on a dataset to maintain a given variance fraction.

    Parameters:
    - X: np.ndarray of shape (n, d) with zero mean
    - var: float, variance threshold (0 < var <= 1)

    Returns:
    - W: np.ndarray of shape (d, nd), weights matrix
    """
    # Perform SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute variance explained by each component
    explained_variance = (S ** 2)
    total_variance = np.sum(explained_variance)
    ratio = explained_variance / total_variance
    cumulative = np.cumsum(ratio)

    # Find number of components to reach var threshold
    nd = np.searchsorted(cumulative, var) + 1

    # Return weight matrix
    return Vt[:nd].T
