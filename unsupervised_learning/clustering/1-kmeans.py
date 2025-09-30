#!/usr/bin/env python3
"""K-means clustering algorithm"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialize centroids using uniform distribution within data bounds
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    for _ in range(iterations):
        # Assign clusters (Euclidean distance)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Store old centroids
        C_old = C.copy()

        for i in range(k):
            points = X[clss == i]
            if points.shape[0] == 0:
                # Reinitialize empty cluster
                C[i] = np.random.uniform(low=min_vals, high=max_vals, size=(1, d))
            else:
                C[i] = np.mean(points, axis=0)

        # Convergence check
        if np.allclose(C, C_old):
            break

    return C, clss
