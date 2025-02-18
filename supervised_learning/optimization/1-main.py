#!/usr/bin/env python3

import numpy as np
norm_constants = __import__('0-norm_constants').normalization_constants
normalize = __import__('1-normalize').normalize


if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))

    # Create dataset with 3 features
    X = np.concatenate((a, b, c), axis=1)

    # Get mean and std from Task 0
    m, s = norm_constants(X)

    print("Before Normalization (First 10 Rows):")
    print(X[:10])

    # Normalize the data
    X = normalize(X, m, s)

    print("\nAfter Normalization (First 10 Rows):")
    print(X[:10])

    # Verify mean & std after normalization
    m, s = norm_constants(X)
    print("\nNew Mean (Should be ~0):", m)
    print("New Std Dev (Should be ~1):", s)
