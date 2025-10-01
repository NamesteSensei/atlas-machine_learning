#!/usr/bin/env python3
"""
Task 4 — GMM init: create pi, m, S without loops.
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Init GMM params.

    Args
    ----
    X : np.ndarray, shape (n, d)
        Data array.
    k : int
        Cluster count (positive).

    Returns
    -------
    pi : np.ndarray, shape (k,)
        Prior probs, split evenly.
    m : np.ndarray, shape (k, d)
        Means via kmeans.
    S : np.ndarray, shape (k, d, d)
        Identity cov arrays.
    On invalid input: (None, None, None).
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0):
        return None, None, None
    if (not isinstance(k, int) or k <= 0 or k > X.shape[0]):
        return None, None, None

    n, d = X.shape  # noqa: F841 (n unused, kept clear)

    # priors: even split
    pi = np.full((k,), 1.0 / k)

    # means via kmeans
    m, _clss = kmeans(X, k)
    if m is None:
        return None, None, None

    # cov: identity stack
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
