#!/usr/bin/env python3
"""
This module contains a function to add two 2D matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list of lists): The first 2D matrix.
        mat2 (list of lists): The second 2D matrix.

    Returns:
        list of lists or None: A new matrix with elements added element-wise,
        or None if the matrices are not the same shape.
    """
    if len(mat1) != len(mat2):
        return None
    if any(len(row1) != len(row2) for row1, row2 in zip(mat1, mat2)):
        return None
    return [
        [a + b for a, b in zip(row1, row2)]
        for row1, row2 in zip(mat1, mat2)
    ]
