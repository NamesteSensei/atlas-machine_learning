#!/usr/bin/env python3
"""
Task 5: Probability density function of a multivariate Gaussian.
"""

import numpy as np


def pdf(X, m, S):
    """
    Evaluate PDF of a multivariate Gaussian distribution.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Data points to evaluate.
    m : np.ndarray of shape (d,)
        Mean vector of the distribution.
    S : np.ndarray of shape (d, d)
        Covariance matrix of the distribution.

    Returns
    -------
    P : np.ndarray of shape (n,)
        PDF values for each data point.
        Each value is floored to 1e-300.
    None
        On invalid input.
    """
    # ---- Validate input ----
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0):
        return None
    if (not isinstance(m, np.ndarray) or m.ndim != 1 or
            m.shape[0] != X.shape[1]):
        return None
    if (not isinstance(S, np.ndarray) or S.ndim != 2 or
            S.shape[0] != S.shape[1] or S.shape[0] != X.shape[1]):
        return None

    n, d = X.shape

    # Determinant and inverse of covariance
    det = np.linalg.det(S)
    if det <= 0:
        return None
    inv_S = np.linalg.inv(S)

    # Normalization constant
    norm_const = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)

    # Centered data
    diff = X - m

    # Mahalanobis distances (vectorized, no loops)
    exp_term = np.einsum('ij,jk,ik->i', diff, inv_S, diff)

    # PDF values
    P = norm_const * np.exp(-0.5 * exp_term)

    # Enforce minimum floor at 1e-300
    P = np.maximum(P, 1e-300)

    return P
