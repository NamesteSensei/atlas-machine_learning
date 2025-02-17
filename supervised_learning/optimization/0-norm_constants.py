#!/usr/bin/env python3
"""
Module for computing normalization constants.

This script contains a function that calculates the mean and
standard deviation for each feature in a dataset. It is used
for standardizing input data before training machine learning models.
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the mean and standard deviation for each feature in X.

    Parameters:
    X (numpy.ndarray): Matrix of shape (m, nx) where:
                       - m = number of samples
                       - nx = number of features

    Returns:
    tuple: (mean, standard deviation) for each feature
    """
    mean = np.mean(X, axis=0)  # Compute mean
    std = np.std(X, axis=0)    # Compute standard deviation
    return mean, std
