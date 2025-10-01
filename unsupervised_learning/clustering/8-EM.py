#!/usr/bin/env python3
"""
8-EM.py

Runs the Expectation-Maximization algorithm on a Gaussian Mixture Model (GMM).
"""

import numpy as np
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
        Threshold on the change in log-likelihood to stop early (default 1e-5).
    verbose : bool, optional
        If True, prints the log-likelihood progress every 10 updates
        and at the end.

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
        if not isinstance(k, int) or k <= 0:
            return None, None, None, None, None
        if not isinstance(iterations, int) or iterations <= 0:
            return None, None, None, None, None
        if not isinstance(tol, (float, int)) or tol < 0:
            return None, None, None, None, None
        if not isinstance(verbose, bool):
            return None, None, None, None, None

        # Initialization of parameters
        pi, m, S = initialize(X, k)
        g, log_likelihood = expectation(X, pi, m, S)

        if g is None:
            return None, None, None, None, None

        prev_l = log_likelihood

        for i in range(iterations):
            # Maximization step
            pi, m, S = maximization(X, g)
            if pi is None:
                return None, None, None, None, None

            # Expectation step
            g, log_likelihood = expectation(X, pi, m, S)
            if g is None:
                return None, None, None, None, None

            # Verbose printing
            if verbose and (i % 10 == 0 or i == iterations - 1):
                print(f"Log Likelihood after {i} iterations: "
                      f"{log_likelihood:.5f}")

            # Convergence check
            if abs(log_likelihood - prev_l) <= tol:
                if verbose:
                    print(f"Log Likelihood after {i + 1} iterations: "
                          f"{log_likelihood:.5f}")
                break

            prev_l = log_likelihood

        return pi, m, S, g, log_likelihood

    except Exception:
        return None, None, None, None, None
