#!/usr/bin/env python3
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity

if __name__ == '__main__':
    # Load confusion matrix
    confusion = np.load('confusion.npz')['confusion']

    # Suppress scientific notation for cleaner output
    np.set_printoptions(suppress=True)

    # Compute and print sensitivity values
    print(sensitivity(confusion))
