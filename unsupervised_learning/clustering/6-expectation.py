#!/usr/bin/env python3
"""
Expectation step in the EM algorithm for a Gaussian Mixture Model (GMM).
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the expectation step in the EM algorithm for a GMM.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Data set containing n data points, each with d dimensions.
    pi : np.ndarray of shape (k,)
        Priors (cluster probabilities) for each of the k clusters.
    m : np.ndarray of shape (k, d)
        Mean vectors of each cluster.
    S : np.ndarray of shape (k, d, d)
        Covariance matrices of each cluster.

    Returns
    -------
    g : np.ndarray of shape (k, n)
        Posterior probabilities (responsibilities) for each data
        point belonging to each cluster.
    l : float
        Total log likelihood of the data given the current parameters.
    None, None on failure
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None
        if not isinstance(pi, np.ndarray) or pi.ndim != 1:
            return None, None
        if not isinstance(m, np.ndarray) or m.ndim != 2:
            return None, None
        if not isinstance(S, np.ndarray) or S.ndim != 3:
            return None, None

        n, d = X.shape
        k = pi.shape[0]

        if m.shape != (k, d) or S.shape != (k, d, d):
            return None, None

        # Weighted likelihoods: prior * pdf for each cluster
        like = np.array([pi[i] * pdf(X, m[i], S[i]) for i in range(k)])

        # Total likelihood across clusters
        tot = np.sum(like, axis=0)

        # Normalize to get posterior probabilities (responsibilities)
        g = like / tot

        # Compute total log likelihood
        l = np.sum(np.log(tot))

        return g, l
    except Exception:
        return None, None
