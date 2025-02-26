#!/usr/bin/env python3
import numpy as np
specificity = __import__('3-specificity').specificity

if __name__ == '__main__':
    # Load confusion matrix
    confusion = np.load('confusion.npz')['confusion']

    # Suppress scientific notation for cleaner output
    np.set_printoptions(suppress=True)

    # Compute and print specificity values
    print(specificity(confusion))
