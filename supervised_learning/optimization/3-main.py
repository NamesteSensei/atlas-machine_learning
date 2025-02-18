#!/usr/bin/env python3

import numpy as np
create_mini_batches = __import__('3-mini_batch').create_mini_batches


if __name__ == '__main__':
    # Create dataset
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10]])

    Y = np.array([[11, 12],
                  [13, 14],
                  [15, 16],
                  [17, 18],
                  [19, 20]])

    batch_size = 2

    # Generate mini-batches
    mini_batches = create_mini_batches(X, Y, batch_size)

    # Print mini-batches
    for i, (X_batch, Y_batch) in enumerate(mini_batches):
        print(f"Batch {i+1}:")
        print("X_batch:\n", X_batch)
        print("Y_batch:\n", Y_batch)
        print()
