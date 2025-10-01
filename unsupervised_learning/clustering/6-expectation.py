#!/usr/bin/env python3
"""
6-expectation.py

Expectation step in the EM algorithm of a Gaussian Mixture Model (GMM).
"""

import numpy as np


pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Executes the expectation step in the EM algorithm of a GMM.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Input data with n samples and d features.
    pi : np.ndarray of shape (k,)
        Prior probabilities, one per cluster.
    m : np.ndarray of shape (k, d)
        Mean vectors, one per cluster.
    S : np.ndarray of shape (k, d, d)
        Covariance matrices, one per cluster.

    Returns
    -------
    g : np.ndarray of shape (k, n)
        Posterior probabilities (responsibilities) giving
        the likelihood of each sample belonging to each cluster.
    log_likelihood : float
        Total log-likelihood across all samples.
    None, None
        Returned on failure if inputs are invalid.
    """
    try:
        # Validate inputs and shapes
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

        # Weighted likelihoods: multiply priors with pdf values
        like = np.array([pi[i] * pdf(X, m[i], S[i]) for i in range(k)])
        tot = np.sum(like, axis=0)

        # Normalize to compute posterior probabilities
        g = like / tot

        # Compute total log-likelihood
        log_likelihood = np.sum(np.log(tot))

        return g, log_likelihood
    except Exception:
        return None, None
