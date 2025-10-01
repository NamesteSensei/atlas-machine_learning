#!/usr/bin/env python3
"""
Task 3: optimum_k
Determine the optimum number of clusters using the elbow method
based on intra-cluster variance.
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Parameters
    ----------
    X : np.ndarray
        Shape (n, d) dataset of n points with d dimensions.
    kmin : int
        Minimum number of clusters (must be >= 1).
    kmax : int
        Maximum number of clusters (must be >= kmin).
    iterations : int
        Number of iterations to run k-means.

    Returns
    -------
    results : list of tuples
        Each tuple is (C, clss), the centroids and cluster assignments
        for each tested k.
    d_vars : list of floats
        Variances corresponding to each cluster count.
    or (None, None) if input validation fails.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0):
        return None, None
    n, d = X.shape
    if (not isinstance(kmin, int) or kmin < 1 or kmin >= n):
        return None, None
    if kmax is None:
        kmax = n
    if (not isinstance(kmax, int) or kmax <= kmin or kmax > n):
        return None, None
    if (not isinstance(iterations, int) or iterations <= 0):
        return None, None

    results, d_vars = [], []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        var = variance(X, C)
        results.append((C, clss))
        d_vars.append(var)

    return results, d_vars
