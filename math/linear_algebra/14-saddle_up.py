#!/usr/bin/env python3

"""
This module performs matrix multiplication on two NumPy arrays.
"""

import numpy as np  # Import placed correctly at the top of the file


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication on two matrices.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.

    Returns:
        numpy.ndarray: The result of matrix multiplication.
    """
    return np.matmul(mat1, mat2)
