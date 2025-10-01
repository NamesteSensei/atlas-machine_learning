#!/usr/bin/env python3
"""
8-EM.py

Runs the Expectation-Maximization algorithm on a Gaussian Mixture Model (GMM).
"""

import numpy as np
import numbers
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Executes the Expectation-Maximization procedure on a GMM.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Input data with n samples and d features.
    k : int
        Number of clusters.
    iterations : int, optional
        Maximum number of update cycles (default 1000).
    tol : float, optional
        Convergence threshold for log-likelihood (default 1e-5).
    verbose : bool, optional
        If True, prints log-likelihood progression.

    Returns
    -------
    pi : np.ndarray of shape (k,)
        Cluster prior probabilities.
    m : np.ndarray of shape (k, d)
        Cluster centroids.
    S : np.ndarray of shape (k, d, d)
        Cluster covariance matrices.
    g : np.ndarray of shape (k, n)
        Posterior probabilities of samples assigned to clusters.
    l : float
        Final log-likelihood value.
    None, None, None, None, None
        Returned on failure if inputs are invalid.
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None, None, None, None
        if not isinstance(k, numbers.Integral) or k <= 0:
            return None, None, None, None, None
        if not isinstance(iterations, numbers.Integral) or iterations <= 0:
            return None, None, None, None, None
        if not isinstance(tol, numbers.Real) or tol < 0:
            return None, None, None, None, None
        if not isinstance(verbose, (bool, np.bool_)):
            return None, None, None, None, None

        # Initialize parameters
        pi, m, S = initialize(X, k)
        g, log_likelihood = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        if verbose:
            print(f"Log Likelihood after 0 iterations: {log_likelihood:.5f}")

        prev_l = log_likelihood

        for i in range(iterations):
            pi, m, S = maximization(X, g)
            if pi is None:
                return None, None, None, None, None

            g, log_likelihood = expectation(X, pi, m, S)
            if g is None:
                return None, None, None, None, None

            if verbose and ((i + 1) % 10 == 0 or i == iterations - 1):
                print(f"Log Likelihood after {i + 1} iterations: "
                      f"{log_likelihood:.5f}")

            if abs(log_likelihood - prev_l) <= tol:
                if verbose:
                    print(f"Log Likelihood after {i + 1} iterations: "
                          f"{log_likelihood:.5f}")
                break

            prev_l = log_likelihood

        return pi, m, S, g, log_likelihood

    except Exception:
        return None, None, None, None, None
