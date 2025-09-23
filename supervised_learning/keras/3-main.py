#!/usr/bin/env python3
"""
3-main.py
---------
Test file for Task 3: One-hot encoding of labels.
"""

import numpy as np
one_hot = __import__('3-one_hot').one_hot

if __name__ == '__main__':
    labels = np.load('MNIST.npz')['Y_train'][:10]
    print(labels)
    print(one_hot(labels))
