#!/usr/bin/env python3
"""
Expectation step for the EM algorithm of a GMM.
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the expectation step in the EM algorithm for a GMM.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        The data set.
    pi : np.ndarray of shape (k,)
        The priors for each cluster.
    m : np.ndarray of shape (k, d)
        The centroid means for each cluster.
    S : np.ndarray of shape (k, d, d)
        The covariance matrices for each cluster.

    Returns
    -------
    g : np.ndarray of shape (k, n)
        The posterior probabilities (responsibilities).
    l : float
        The total log likelihood.
    None, None on failure.
    """
    # Validate X
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0):
        return None, None
    # Validate pi
    if (not isinstance(pi, np.ndarray) or pi.ndim != 1 or
            pi.size == 0 or not np.isclose(np.sum(pi), 1)):
        return None, None
    # Validate m
    if (not isinstance(m, np.ndarray) or m.ndim != 2 or
            m.shape[0] != pi.shape[0] or m.shape[1] != X.shape[1]):
        return None, None
    # Validate S
    if (not isinstance(S, np.ndarray) or S.ndim != 3 or
            S.shape[0] != pi.shape[0] or
            S.shape[1] != X.shape[1] or S.shape[2] != X.shape[1]):
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    # Allocate space for PDFs
    pdfs = np.zeros((k, n))

    # --- Only allowed loop: compute PDF for each cluster
    for i in range(k):
        pdfs[i] = pdf(X, m[i], S[i])

    # Weighted likelihoods
    weighted = pi[:, None] * pdfs
    total = np.sum(weighted, axis=0)

    # Normalize to responsibilities
    g = weighted / total
    likelihood = np.sum(np.log(total))

    return g, likelihood
