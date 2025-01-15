#!/usr/bin/env python3
"""
This module contains a function to transpose a NumPy array.
"""


def np_transpose(matrix):
    """
    Transposes a numpy.ndarray.

    Args:
        matrix (numpy.ndarray): The input array to transpose.

    Returns:
        numpy.ndarray: A new array that is the transpose of the input matrix.
    """
    return matrix.T
