#!/usr/bin/env python3
"""
Main file to test expectation step of Gaussian Mixture Models.
Covers normal case and all validation error cases.
"""

import numpy as np
expectation = __import__('6-expectation').expectation
initialize = __import__('4-initialize').initialize


if __name__ == "__main__":
    np.random.seed(1)

    # Generate synthetic data (Normal case)
    X = np.random.multivariate_normal([10, 5], [[3, 1], [1, 4]], size=50)
    pi, m, S = initialize(X, 2)
    g, likelihood = expectation(X, pi, m, S)
    print("Normal")
    print(g.shape, likelihood)

    # ---- Error cases ----
    # Invalid X
    g, likelihood = expectation("not an array", pi, m, S)
    print("invalid X:", g, likelihood)

    # High dimensional X (not 2D)
    g, likelihood = expectation(np.array([1, 2, 3]), pi, m, S)
    print("High dimensional X:", g, likelihood)

    # Invalid pi
    bad_pi = np.array([0.6, 0.3])  # does not sum to 1
    g, likelihood = expectation(X, bad_pi, m, S)
    print("invalid pi:", g, likelihood)

    # Invalid m
    bad_m = np.array([1, 2, 3])  # not 2D
    g, likelihood = expectation(X, pi, bad_m, S)
    print("invalid m:", g, likelihood)

    # Invalid S
    bad_S = np.array([[[1, 0], [0, 1]]])  # only 1 covariance
    g, likelihood = expectation(X, pi, m, bad_S)
    print("invalid S:", g, likelihood)
