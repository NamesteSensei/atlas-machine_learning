#!/usr/bin/env python3
import numpy as np


def variance(X, C):
    """
    Calculate total intra-cluster variance.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data matrix containing n points with d features.
    C : np.ndarray, shape (k, d)
        Centroid matrix containing k cluster centers.

    Returns
    -------
    numpy.float64
        Sum of squared distances from each point to its nearest centroid.
        Returns None on invalid input.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0):
        return None
    if (not isinstance(C, np.ndarray) or C.ndim != 2 or
            C.shape[1] != X.shape[1] or C.size == 0):
        return None

    diffs = X[:, None, :] - C[None, :, :]
    dists = np.sum(diffs * diffs, axis=2)

    return np.sum(np.min(dists, axis=1))
