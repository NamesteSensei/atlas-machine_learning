#!/usr/bin/env python3
"""
K-means clustering algorithm implementation.
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Parameters:
    - X (np.ndarray): shape (n, d), dataset
    - k (int): number of clusters
    - iterations (int): maximum number of iterations

    Returns:
    - C (np.ndarray): shape (k, d), centroid means for each cluster
    - clss (np.ndarray): shape (n,), index of the cluster each point belongs to
    - (None, None) if input is invalid
    """
    # --- Input validation ---
    if (type(X) is not np.ndarray or X.ndim != 2 or X.size == 0):
        return None, None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    # --- Initialization ---
    n, d = X.shape
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    for _ in range(iterations):  # loop 1
        # Assign clusters (distance matrix n×k)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids (loop 2)
        new_C = np.copy(C)
        for i in range(k):
            if np.any(clss == i):
                new_C[i] = X[clss == i].mean(axis=0)
            else:  # reinitialize empty cluster
                new_C[i] = np.random.uniform(low=min_vals, high=max_vals)

        # Early stopping if converged
        if np.allclose(C, new_C):
            C = new_C
            break
        C = new_C

    return C, clss
