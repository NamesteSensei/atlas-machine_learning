#!/usr/bin/env python3
"""
K-means clustering algorithm implementation.
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        The dataset.
    k : int
        Number of clusters (k > 0 and k <= n).
    iterations : int
        Maximum number of iterations (iterations > 0).

    Returns
    -------
    C : np.ndarray of shape (k, d)
        Final centroid means for each cluster.
    clss : np.ndarray of shape (n,)
        Index of the cluster in C that each data point belongs to.

    Notes
    -----
    - Uses multivariate uniform initialization within per-dimension min/max.
    - If a cluster becomes empty, its centroid is reinitialized by drawing
      new coordinates uniformly within those per-dimension bounds.
    - At most two loops are used: the outer EM loop and a small loop over k
      to compute new means (vectorization elsewhere).
    - On invalid input, returns (None, None).
    """
    # ---- Validation ----
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    # ---- Initialize centroids (uniform once) ----
    C = np.random.uniform(low=mins, high=maxs, size=(k, d))

    # ---- K-means iterations (loop 1) ----
    for _ in range(iterations):
        # Assign points to nearest centroid (vectorized)
        # Use squared distances to avoid unnecessary sqrt
        # distances: (n, k)
        diffs = X[:, None, :] - C[None, :, :]
        distances = np.sum(diffs * diffs, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids (loop 2 over k)
        new_C = C.copy()
        for i in range(k):
            mask = (clss == i)
            if np.any(mask):
                new_C[i] = X[mask].mean(axis=0)
            # else: handled in vector step below

        # Handle all empty clusters at once with a single uniform call
        empty = ~np.isin(np.arange(k), clss)
        if np.any(empty):
            # One uniform call for all empties keeps RNG usage consistent
            new_C[empty] = np.random.uniform(low=mins, high=maxs,
                                             size=(np.sum(empty), d))

        # Early stop if centroids no longer change (exact comparison)
        # Using exact equality is safe here because means stabilize.
        if np.all(C == new_C):
            C = new_C
            break

        C = new_C

    # Recompute final assignments so clss matches the returned C
    diffs = X[:, None, :] - C[None, :, :]
    distances = np.sum(diffs * diffs, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
