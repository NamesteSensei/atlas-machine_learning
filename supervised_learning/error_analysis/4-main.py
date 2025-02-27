#!/usr/bin/env python3
import numpy as np
f1_score = __import__('4-f1_score').f1_score

if __name__ == '__main__':
    # Load confusion matrix
    confusion = np.load('confusion.npz')['confusion']

    # Suppress scientific notation for cleaner output
    np.set_printoptions(suppress=True)

    # Compute and print F1 scores
    print(f1_score(confusion))
