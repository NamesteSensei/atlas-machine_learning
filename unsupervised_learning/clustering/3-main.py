#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

optimum_k = __import__('3-optimum').optimum_k

if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    results, d_vars = optimum_k(X, kmin=1, kmax=10)
    for k, var in enumerate(d_vars, 1):
        print("Variance for {} clusters: {:.2f}".format(k, var))

    # Elbow plot
    plt.plot(range(1, 11), d_vars, 'o-')
    plt.xlabel("k")
    plt.ylabel("Variance")
    plt.title("Elbow Method")
    plt.show()
