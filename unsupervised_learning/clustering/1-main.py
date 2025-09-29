#!/usr/bin/env python3
"""
Test script for kmeans function.
"""

import numpy as np
import matplotlib.pyplot as plt
kmeans = __import__('1-kmeans').kmeans


if __name__ == "__main__":
    np.random.seed(0)

    # Create synthetic clusters
    a = np.random.multivariate_normal(
        [30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal(
        [10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal(
        [40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal(
        [60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal(
        [20, 70], [[16, 0], [0, 16]], size=50)

    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    # Run K-means with 5 clusters
    C, clss = kmeans(X, 5)

    print(C)

    # Visualize
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(C[:, 0], C[:, 1], s=50,
                marker='*', c=list(range(5)))
    plt.show()
