#!/usr/bin/env python3
"""
This module performs matrix multiplication on two NumPy arrays.
"""


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication on two matrices.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.

    Returns:
        numpy.ndarray: The result of matrix multiplication.
    """
    import numpy as np
    return np.matmul(mat1, mat2)
