#!/usr/bin/env python3
"""
9-main.py

Test driver for the BIC-based selection of the number of clusters in a GMM.
"""

import matplotlib.pyplot as plt
import numpy as np
BIC = __import__('9-BIC').BIC


if __name__ == '__main__':
    np.random.seed(11)

    # Build synthetic Gaussian clusters
    a = np.random.multivariate_normal([30, 40],
                                      [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25],
                                      [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30],
                                      [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70],
                                      [[35, 10], [10, 35]], size=1000)

    # Merge into a single dataset
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)

    # Run BIC evaluation
    best_k, best_result, l, b = BIC(X, kmin=1, kmax=10)

    # Print outputs
    print(best_k)
    print(best_result)
    print(l)
    print(b)

    # Plot log-likelihood progression
    x = np.arange(1, 11)
    plt.plot(x, l, 'r')
    plt.xlabel('Clusters')
    plt.ylabel('Log Likelihood')
    plt.tight_layout()
    plt.show()

    # Plot BIC values
    plt.plot(x, b, 'b')
    plt.xlabel('Clusters')
    plt.ylabel('BIC')
    plt.tight_layout()
    plt.show()
