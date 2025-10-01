#!/usr/bin/env python3
"""
6-expectation.py

Performs the Expectation step of the EM algorithm
for a Gaussian Mixture Model (GMM).
"""

import numpy as np


pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the expectation step in the EM algorithm for a GMM.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Dataset with n data points of d dimensions each.
    pi : np.ndarray of shape (k,)
        Priors (cluster probabilities) for each of the k clusters.
    m : np.ndarray of shape (k, d)
        Mean vectors (centroids) of the clusters.
    S : np.ndarray of shape (k, d, d)
        Covariance matrices of the clusters.

    Returns
    -------
    g : np.ndarray of shape (k, n)
        Posterior probabilities (responsibilities) of clusters
        for each data point.
    log_likelihood : float
        Total log likelihood of the data under the current model.
    None, None
        On failure due to invalid input.
    """
    try:
        # Validate input types and dimensions
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

        # Compute weighted likelihoods: pi * pdf for each cluster
        like = np.array([pi[i] * pdf(X, m[i], S[i]) for i in range(k)])
        tot = np.sum(like, axis=0)

        # Normalize to get posterior probabilities
        g = like / tot

        # Compute total log likelihood
        log_likelihood = np.sum(np.log(tot))

        return g, log_likelihood
    except Exception:
        return None, None
