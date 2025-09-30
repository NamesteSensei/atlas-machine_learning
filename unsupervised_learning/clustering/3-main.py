#!/usr/bin/env python3
"""
Main file to test Task 3: optimum_k
"""

import numpy as np
optimum_k = __import__('3-optimum').optimum_k


if __name__ == "__main__":
    np.random.seed(0)

    # ---- Build a simple dataset ----
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=30)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=30)
    X = np.concatenate((a, b), axis=0)

    # ---- Normal case ----
    results, d_vars = optimum_k(X, kmin=1, kmax=5)
    print("Normal")
    print("Number of results:", len(results))
    print("Variances:", [round(v, 2) for v in d_vars])

    # ---- High dimensional X ----
    X_high = np.random.rand(50, 5)   # 5D data
    results, d_vars = optimum_k(X_high, kmin=1, kmax=3)
    print("High dimensional X")
    print(results is not None and d_vars is not None)

    # ---- Invalid X (not array) ----
    results, d_vars = optimum_k("not an array", kmin=1, kmax=3)
    print("invalid X:", results, d_vars)

    # ---- Invalid kmin (too small) ----
    results, d_vars = optimum_k(X, kmin=0, kmax=3)
    print("invalid kmin:", results, d_vars)

    # ---- Invalid kmax (too large) ----
    results, d_vars = optimum_k(X, kmin=1, kmax=999)
    print("invalid kmax:", results, d_vars)

    # ---- kmax <= kmin ----
    results, d_vars = optimum_k(X, kmin=5, kmax=3)
    print("kmax >= kmin:", results, d_vars)

    # ---- Invalid iterations ----
    results, d_vars = optimum_k(X, kmin=1, kmax=3, iterations=0)
    print("invalid iterations:", results, d_vars)
