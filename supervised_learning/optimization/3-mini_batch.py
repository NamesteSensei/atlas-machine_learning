#!/usr/bin/env python3
"""
Module for creating mini-batches from a dataset.

Mini-batches are used in machine learning to train models
using mini-batch gradient descent.
"""

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches from a dataset for mini-batch gradient descent.

    Parameters:
    X (numpy.ndarray): Data of shape (m, nx) where:
        - m is the number of data points
        - nx is the number of features
    Y (numpy.ndarray): Labels of shape (m, ny) where:
        - ny is the number of labels/classes
    batch_size (int): Number of samples per batch.

    Returns:
    list: A list of mini-batches, each containing a tuple (X_batch, Y_batch).
    """
    # Shuffle X and Y together
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []

    # Create mini-batches
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
