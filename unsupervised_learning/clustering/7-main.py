#!/usr/bin/env python3
"""
7-main.py

Test script for the maximization step of the EM algorithm.
"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


if __name__ == '__main__':
    np.random.seed(11)

    # Generate synthetic Gaussian clusters
    a = np.random.multivariate_normal([30, 40],
                                      [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25],
                                      [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30],
                                      [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70],
                                      [[35, 10], [10, 35]], size=1000)

    # Combine all clusters into dataset
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)

    # Initialize parameters
    pi, m, S = initialize(X, 4)

    # E-step: compute responsibilities
    g, _ = expectation(X, pi, m, S)

    # M-step: update parameters
    pi, m, S = maximization(X, g)

    # Print results
    print(pi)
    print(m)
    print(S)
