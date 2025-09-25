#!/usr/bin/env python3
import numpy as np

def pca(X, var=0.95):
    """
    Performs PCA on a dataset to maintain a certain amount of variance.

    Parameters:
    - X: np.ndarray of shape (n, d) with mean already zeroed
    - var: float (0 < var <= 1), target fraction of variance to preserve

    Returns:
    - W: np.ndarray of shape (d, nd), projection matrix
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    variances = S**2
    cumulative_variance = np.cumsum(variances) / np.sum(variances)
    nd = np.searchsorted(cumulative_variance, var) + 1
    return Vt[:nd].T
