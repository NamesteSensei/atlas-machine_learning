#!/usr/bin/env python3
"""
Expectation step for Gaussian Mixture Models (GMM).
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the expectation step in the EM algorithm for a GMM.

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        Data set.
    pi : numpy.ndarray of shape (k,)
        Priors for each cluster.
    m : numpy.ndarray of shape (k, d)
        Centroid means for each cluster.
    S : numpy.ndarray of shape (k, d, d)
        Covariance matrices for each cluster.

    Returns
    -------
    g : numpy.ndarray of shape (k, n)
        Posterior probabilities (responsibilities).
    likelihood : float
        Total log likelihood of X under the model.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0):
        return None, None
    if (not isinstance(pi, np.ndarray) or pi.ndim != 1 or
            not np.isclose(np.sum(pi), 1)):
        return None, None
    if (not isinstance(m, np.ndarray) or m.ndim != 2 or
            m.shape[0] != pi.shape[0] or m.shape[1] != X.shape[1]):
        return None, None
    if (not isinstance(S, np.ndarray) or S.ndim != 3 or
            S.shape[0] != pi.shape[0] or
            S.shape[1] != X.shape[1] or S.shape[2] != X.shape[1]):
        return None, None

    k = pi.shape[0]

    # Compute PDFs for all clusters
    pdfs = np.vstack([
        pdf(X, m[i], S[i])
        for i in range(k)
    ])

    # Weighted responsibilities
    weighted = pi[:, None] * pdfs
    total = np.sum(weighted, axis=0)

    # Normalize to get posterior probabilities
    g = weighted / total
    likelihood = np.sum(np.log(total))

    return g, likelihood
