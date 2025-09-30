#!/usr/bin/env python3
"""
Compute the total intra-cluster variance for a dataset.
"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Dataset of n points with d features.
    C : np.ndarray of shape (k, d)
        Centroid means for each cluster.

    Returns
    -------
    var : float
        Total intra-cluster variance.
    None
        If inputs are invalid.
    """
    # ---- Validation ----
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0):
        return None
    if (not isinstance(C, np.ndarray) or C.ndim != 2 or
            C.shape[1] != X.shape[1] or C.size == 0):
        return None

    # ---- Compute squared distances (n, k) ----
    # Broadcasting: expand X and C to compare all points vs. all centroids
    diffs = X[:, None, :] - C[None, :, :]
    dists = np.sum(diffs ** 2, axis=2)

    # ---- Take min distance for each point, sum them ----
    var = float(np.sum(np.min(dists, axis=1)))

    return var
