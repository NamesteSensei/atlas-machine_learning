#!/usr/bin/env python3
import numpy as np
precision = __import__('2-precision').precision

if __name__ == '__main__':
    # Load confusion matrix
    confusion = np.load('confusion.npz')['confusion']

    # Suppress scientific notation for cleaner output
    np.set_printoptions(suppress=True)

    # Compute and print precision values
    print(precision(confusion))
