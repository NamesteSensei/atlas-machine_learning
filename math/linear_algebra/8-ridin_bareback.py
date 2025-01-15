#!/usr/bin/env python3
"""
This module contains a function to perform matrix multiplication.

Matrix multiplication is when we match rows from the first matrix
with columns from the second matrix to create a new matrix.
"""

import numpy as np  # Importing NumPy for handling numbers and matrices.


def mat_mul(mat1, mat2):
    """
    Multiplies two matrices and returns the result.

    Args:
        mat1 (list of lists): The first matrix.
        mat2 (list of lists): The second matrix.

    Returns:
        list of lists or None: The resulting matrix after multiplication,
                               or None if the matrices cannot be multiplied.
    """
    try:
        # Use NumPy to multiply the matrices.
        return np.matmul(mat1, mat2).tolist()
    except ValueError:
        # If multiplication isn't possible (sizes don't match), return None.
        return None
