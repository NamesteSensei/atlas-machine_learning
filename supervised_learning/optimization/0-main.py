#!/usr/bin/env python3

import numpy as np
normalization_constants = __import__('0-norm_constants').normalization_constants

# Test the normalization_constants function
if __name__ == '__main__':
    np.random.seed(0)
    a = np.random.normal(0, 2, size=(100, 1))
    b = np.random.normal(2, 1, size=(100, 1))
    c = np.random.normal(-3, 10, size=(100, 1))

    # Create a dataset with 3 features
    X = np.concatenate((a, b, c), axis=1)

    # Get mean and std
    m, s = normalization_constants(X)
    
    # Print results
    print("Mean:", m)
    print("Standard Deviation:", s)
