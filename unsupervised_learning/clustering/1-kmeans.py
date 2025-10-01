#!/usr/bin/env python3
"""
K-means clustering algorithm from scratch.
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on dataset X.

    Args:
        X: ndarray of shape (n, d)
        k: number of clusters
        iterations: max number of updates

    Returns:
        C: ndarray (k, d) - centroid means
        clss: ndarray (n,) - cluster assignments
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(min_vals, max_vals, (k, d))

    for _ in range(iterations):
        D = np.linalg.norm(X[:, None] - C[None, :], axis=2)
        clss = np.argmin(D, axis=1)

        new_C = np.copy(C)
        for i in range(k):
            points = X[clss == i]
            if points.shape[0] == 0:
                new_C[i] = np.random.uniform(min_vals, max_vals)
            else:
                new_C[i] = np.mean(points, axis=0)

        if np.allclose(C, new_C):
            break
        C = new_C

    return C, clss
