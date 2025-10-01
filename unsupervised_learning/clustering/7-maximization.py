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
        # Validate X
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None, None
        # Validate g
        if not isinstance(g, np.ndarray) or g.ndim != 2:
            return None, None, None

        n, d = X.shape
        k, n_check = g.shape

        # Shape mismatch
        if n != n_check:
            return None, None, None

        # Validate range of g (allow tiny numerical noise)
        if np.any(g < -1e-8) or np.any(g > 1 + 1e-8):
            return None, None, None

        # Validate that responsibilities sum to 1 per sample
        if not np.allclose(np.sum(g, axis=0), 1, atol=1e-6):
            return None, None, None

        # Effective number of samples assigned to each cluster
        Nk = np.sum(g, axis=1)

        # Updated priors
        pi = Nk / n

        # Updated means
        m = (g @ X) / Nk[:, np.newaxis]

        # Updated covariance matrices
        S = np.zeros((k, d, d))
        for i in range(k):
            diff = X - m[i]
            weighted = g[i][:, np.newaxis] * diff
            S[i] = (weighted.T @ diff) / Nk[i]

        return pi, m, S

    except Exception:
        return None, None, None
