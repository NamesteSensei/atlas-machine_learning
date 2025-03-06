#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot

if __name__ == '__main__':
    # Labels as "tickets" to different sections
    labels = np.array([1, 3, 4, 0])
    classes = 5
    one_hot_matrix = one_hot(labels, classes)
    print(one_hot_matrix)

    # Expected Output:
    # [[0. 1. 0. 0. 0.]  # Section B glowing
    #  [0. 0. 0. 1. 0.]  # Section D glowing
    #  [0. 0. 0. 0. 1.]  # Section E glowing
    #  [1. 0. 0. 0. 0.]] # Section A glowing
