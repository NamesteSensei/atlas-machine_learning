#!/usr/bin/env python3
"""
Task 4: Initialize Gaussian Mixture Model parameters.
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model (GMM).

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n, d), where n is the number of data points
        and d is the number of dimensions.
    k : int
        Positive integer representing the number of clusters.

    Returns
    -------
    pi : np.ndarray of shape (k,)
        Priors for each cluster, initialized evenly.
    m : np.ndarray of shape (k, d)
        Centroid means for each cluster, initialized using k-means.
    S : np.ndarray of shape (k, d, d)
        Covariance matrices for each cluster, initialized as identity matrices.
    Or (None, None, None) on failure.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0):
        return None, None, None
    if (not isinstance(k, int) or k <= 0 or k > X.shape[0]):
        return None, None, None

    n, d = X.shape

    # Priors: evenly split across clusters
    pi = np.full((k,), 1 / k)

    # Means: initialized using k-means
    m, _ = kmeans(X, k)

    if m is None:
        return None, None, None

    # Covariances: identity matrices for each cluster
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
