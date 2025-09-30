#!/usr/bin/env python3
"""K-means clustering algorithm"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset"""
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # First use of np.random.uniform → initialization
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    for _ in range(iterations):
        # Assign each point to nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        C_new = np.copy(C)

        for i in range(k):  # loop 1
            cluster_points = X[clss == i]
            if cluster_points.size == 0:
                # Deterministic: pick a random existing point
                idx = np.random.randint(0, n)
                C_new[i] = X[idx]
            else:
                C_new[i] = np.mean(cluster_points, axis=0)

        # Early stop if no change
        if np.allclose(C, C_new):
            break

        C = C_new

    return C, clss
