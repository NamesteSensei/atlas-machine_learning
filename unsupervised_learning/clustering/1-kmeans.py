#!/usr/bin/env python3
"""
K-means clustering algorithm implementation.
"""

import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Parameters:
    - X (np.ndarray): shape (n, d), data points
    - k (int): number of clusters
    - iterations (int): max number of iterations

    Returns:
    - C (np.ndarray): shape (k, d), final centroids
    - clss (np.ndarray): shape (n,), index of cluster for each point
    """
    if (type(X) is not np.ndarray or len(X.shape) != 2 or
            type(k) is not int or k <= 0 or
            type(iterations) is not int or iterations <= 0):
        return None, None

    C = initialize(X, k)  # initialize centroids
    for _ in range(iterations):
        # compute distances (n x k)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # update centroids
        new_C = np.array([X[clss == i].mean(axis=0) if np.any(clss == i)
                          else initialize(X, 1) for i in range(k)])

        # stop if converged
        if np.allclose(C, new_C):
            break
        C = new_C

    return C, clss
