#!/usr/bin/env python3
"""
Module for normalizing a dataset using mean and standard deviation.

This script provides a function to standardize features in a dataset.
Normalization ensures that features have a mean of 0 and std of 1.
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a dataset using mean and standard deviation.

    Parameters:
    X (numpy.ndarray): Matrix of shape (d, nx) where:
        - d is the number of data points
        - nx is the number of features
    m (numpy.ndarray): Array of shape (nx,) containing the mean of features.
    s (numpy.ndarray): Array of shape (nx,) containing the std of features.

    Returns:
    numpy.ndarray: The normalized X matrix.
    """
    return (X - m) / s
