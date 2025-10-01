#!/usr/bin/env python3
"""
7-maximization.py

Maximization step in the EM algorithm of a Gaussian Mixture Model (GMM).
"""

import numpy as np


def maximization(X, g):
    """
    Executes the maximization step in the EM algorithm of a GMM.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Input data with n samples and d features.
    g : np.ndarray of shape (k, n)
        Posterior probabilities (responsibilities).

    Returns
    -------
    pi : np.ndarray of shape (k,)
        Updated priors.
    m : np.ndarray of shape (k, d)
        Updated mean vectors.
    S : np.ndarray of shape (k, d, d)
        Updated covariance matrices.
    None, None, None
        Returned on failure if inputs are invalid.
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None, None
        if not isinstance(g, np.ndarray) or g.ndim != 2:
            return None, None, None

        n, d = X.shape
        k, n_check = g.shape

        # Shape must align
        if n != n_check:
            return None, None, None

        # Responsibilities must be between 0 and 1
        if np.any(g < 0) or np.any(g > 1):
            return None, None, None

        # Columns of g should sum to 1 (within tolerance)
        if not np.allclose(np.sum(g, axis=0), 1):
            return None, None, None

        # Effective counts
        Nk = np.sum(g, axis=1)

        # Updated priors
        pi = Nk / n

        # Updated means
        m = (g @ X) / Nk[:, np.newaxis]

        # Updated covariances
        S = np.zeros((k, d, d))
        for i in range(k):
            diff = X - m[i]
            weighted = g[i][:, np.newaxis] * diff
            S[i] = (weighted.T @ diff) / Nk[i]

        return pi, m, S
    except Exception:
        return None, None, None
