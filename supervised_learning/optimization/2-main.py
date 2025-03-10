#!/usr/bin/env python3

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


if __name__ == '__main__':
    # Define dataset
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

    # Set seed for reproducibility
    np.random.seed(0)

    # Shuffle data
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    # Print shuffled data
    print("Shuffled X:")
    print(X_shuffled)

    print("\nShuffled Y:")
    print(Y_shuffled)
