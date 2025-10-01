#!/usr/bin/env python3
"""
9-BIC.py

Selects the optimal number of clusters in a GMM using
the Bayesian Information Criterion (BIC).
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Determines the best number of clusters in a GMM using BIC.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Input data with n samples and d features.
    kmin : int, optional
        Minimum number of clusters to check (inclusive).
    kmax : int, optional
        Maximum number of clusters to check (inclusive).
        If None, defaults to n.
    iterations : int, optional
        Maximum number of update cycles in EM (default 1000).
    tol : float, optional
        Convergence threshold in EM (default 1e-5).
    verbose : bool, optional
        If True, EM prints log-likelihood values.

    Returns
    -------
    best_k : int
        Optimal cluster count chosen by BIC.
    best_result : tuple
        (pi, m, S) parameters for the optimal model.
    l : np.ndarray
        Log-likelihood values for each k tested.
    b : np.ndarray
        BIC values for each k tested.
    None, None, None, None
        Returned on failure if inputs are invalid.
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None, None, None
        if not isinstance(kmin, int) or kmin <= 0:
            return None, None, None, None
        if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
            return None, None, None, None
        if not isinstance(iterations, (int, np.integer)) or iterations <= 0:
            return None, None, None, None
        if not isinstance(tol, (float, int, np.floating)) or tol < 0:
            return None, None, None, None
        if not isinstance(verbose, (bool, np.bool_)):
            return None, None, None, None

        n, d = X.shape
        if kmax is None:
            kmax = n
        if kmin > kmax:
            return None, None, None, None

        ks = np.arange(kmin, kmax + 1)
        l_vals, b_vals, models = [], [], []

        for k in ks:
            pi, m, S, g, log_likelihood = expectation_maximization(
                X, k, iterations, tol, verbose
            )

            if pi is None:
                l_vals.append(-np.inf)
                b_vals.append(np.inf)
                models.append((None, None, None))
                continue

            # Parameters count
            p = (k * d) + (k * d * (d + 1) / 2) + (k - 1)
            bic = p * np.log(n) - 2 * log_likelihood

            l_vals.append(log_likelihood)
            b_vals.append(bic)
            models.append((pi, m, S))

        l = np.array(l_vals)
        b = np.array(b_vals)

        best_idx = np.argmin(b)
        best_k = ks[best_idx]
        best_result = models[best_idx]

        return best_k, best_result, l, b

    except Exception:
        return None, None, None, None
