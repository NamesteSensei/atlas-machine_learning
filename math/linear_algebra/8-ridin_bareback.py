#!/usr/bin/env python3
"""
This module contains a function to perform matrix multiplication.

Matrix multiplication is when we match rows from the first matrix
with columns from the second matrix to create a new matrix.
"""


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
    # Check if multiplication is possible (columns of mat1 == rows of mat2)
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Perform the multiplication
    for i in range(len(mat1)):  # Rows of mat1
        for j in range(len(mat2[0])):  # Columns of mat2
            for k in range(len(mat2)):  # Rows of mat2
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
