#!/usr/bin/env python3
"""
Main file for Task 3
Runs optimum_k on a generated dataset and prints variances.
"""

import numpy as np
optimum_k = __import__('3-optimum').optimum_k

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)

    results, d_vars = optimum_k(X, kmin=1, kmax=10, iterations=1000)

    if results is None:
        print("Invalid input")
    else:
        for k, var in enumerate(d_vars, start=1):
            print("Variance with {} clusters: {:.5f}".format(k, var))
