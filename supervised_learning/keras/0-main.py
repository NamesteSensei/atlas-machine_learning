#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot

if __name__ == '__main__':
    # Generating a larger set of labels to match the expected output
    labels = np.array([8, 0, 10, 11, 9, 10, 6, 0, 12, 7, 14, 17, 2, 2, 1, 5, 8,
                       14, 1, 10, 7, 11, 1, 15, 16, 5, 17, 14, 0, 0, 9, 5, 7, 5,
                       14, 1, 17, 1, 10, 7, 11, 4, 3, 16, 16, 0, 17, 11, 0, 13,
                       5, 16, 14, 8, 15, 3, 4, 16, 1, 17, 8, 2, 4, 9, 5, 7, 5,
                       14, 1, 17, 1, 10, 7, 11, 4, 3, 16, 16, 0, 17, 11, 0, 13,
                       5, 16, 14, 8, 15, 3, 4, 16, 1, 17, 8, 2, 4, 9, 5, 7])
    classes = 18  # Matching the expected output's column size
    one_hot_matrix = one_hot(labels, classes)
    print(one_hot_matrix)
