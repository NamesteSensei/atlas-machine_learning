#!/usr/bin/env python3
"""
This module provides a function to determine the optimum number of clusters
for K-means using the elbow method.
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests different cluster counts to determine optimum k.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Dataset of n points with d features.
    kmin : int, optional
        Minimum number of clusters to test (default=1).
    kmax : int, optional
        Maximum number of clusters to test (default=None → n).
    iterations : int, optional
        Maximum number of K-means iterations (default=1000).

    Returns
    -------
    results : list of tuples
        Each element is (C, clss) for a given k.
    d_vars : list of floats
        Variances corresponding to each k.
    None, None
        If input is invalid.
    """
    # ---- Validation ----
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0):
        return None, None
    n = X.shape[0]
    if (not isinstance(kmin, int) or kmin <= 0 or kmin >= n):
        return None, None
    if kmax is None:
        kmax = n
    if (not isinstance(kmax, int) or kmax <= kmin or kmax > n):
        return None, None
    if (not isinstance(iterations, int) or iterations <= 0):
        return None, None

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        d_vars.append(variance(X, C))

    return results, d_vars
