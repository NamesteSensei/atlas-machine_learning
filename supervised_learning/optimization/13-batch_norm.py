#!/usr/bin/env python3
"""
Module for implementing Batch Normalization in NumPy.
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon=1e-7):
    """
    Normalizes an unactivated output of a neural network using batch norm.

    Parameters:
    Z (numpy.ndarray): Shape (m, n), where m is batch size, n is features.
    gamma (numpy.ndarray): Scale factor (1, n).
    beta (numpy.ndarray): Offset factor (1, n).
    epsilon (float): Small number to avoid division by zero.

    Returns:
    numpy.ndarray: Normalized output.
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    # Normalize Z using mean and variance
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    # Apply scaling (gamma) and shifting (beta)
    Z_batch_norm = gamma * Z_norm + beta

    return Z_batch_norm
