#!/usr/bin/env python3
"""
Performs K-means clustering on a dataset.
"""

import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Dataset, where n is number of points and d is dimensions.
    k : int
        Number of clusters.
    iterations : int, optional
        Maximum number of iterations (default is 1000).

    Returns
    -------
    C : np.ndarray of shape (k, d) or None
        Centroid means for each cluster.
    clss : np.ndarray of shape (n,) or None
        Index of the cluster each point belongs to.
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    # Step 1: Initialize centroids
    C = initialize(X, k)
    if C is None:
        return None, None

    for _ in range(iterations):
        # Step 2: Assign each point to nearest centroid
        dist = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        clss = np.argmin(dist, axis=1)

        # Step 3: Update centroids
        new_C = np.array([
            X[clss == j].mean(axis=0) if np.any(clss == j)
            else initialize(X, 1).reshape(-1)
            for j in range(k)
        ])

        # Step 4: Check convergence
        if np.allclose(C, new_C):
            break
        C = new_C

    return C, clss
