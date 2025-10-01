#!/usr/bin/env python3
"""Performs K-means clustering on a dataset."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Executes K-means clustering on input data."""
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(k, int) or k <= 0 or
        not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape
    min_, max_ = np.min(X, axis=0), np.max(X, axis=0)
    C = np.random.uniform(min_, max_, size=(k, d))

    for _ in range(iterations):
        D = np.linalg.norm(X[:, None] - C[None, :], axis=2)
        clss = np.argmin(D, axis=1)

        new_C = np.copy(C)
        i = 0
        while i < k:
            pts = X[clss == i]
            new_C[i] = (np.mean(pts, axis=0) if pts.shape[0] > 0
                        else np.random.uniform(min_, max_))
            i += 1

        if np.allclose(C, new_C):
            break
        C = new_C

    idx = np.lexsort(C.T)
    C = C[idx]
    label_map = np.zeros(k, dtype=int)
    label_map[idx] = np.arange(k)
    clss = label_map[clss]

    return C, clss
