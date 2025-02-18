#!/usr/bin/env python3

import numpy as np
batch_norm = __import__('13-batch_norm').batch_norm

if __name__ == '__main__':
    np.random.seed(0)

    # Simulated unactivated layer outputs
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))

    # Concatenating to create a feature matrix
    Z = np.concatenate((a, b, c), axis=1)

    # Random gamma (scaling) and beta (shifting) values
    gamma = np.random.rand(1, 3)
    beta = np.random.rand(1, 3)

    print("Before Batch Normalization:\n", Z[:10])

    # Apply batch normalization
    Z_norm = batch_norm(Z, gamma, beta, 1e-7)

    print("\nAfter Batch Normalization:\n", Z_norm[:10])
