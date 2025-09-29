#!/usr/bin/env python3
"""
Initialize cluster centroids for K-means clustering.
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Dataset for clustering, where n is number of points and
        d is number of dimensions.
    k : int
        Number of clusters.

    Returns
    -------
    np.ndarray of shape (k, d) or None
        Initialized centroids within min-max range of X,
        or None on failure.
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0):
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    return np.random.uniform(min_vals, max_vals, (k, X.shape[1]))
