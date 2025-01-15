#!/usr/bin/env python3
"""
This module contains a function to concatenate two matrices along
a specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Args:
        mat1 (list of lists): The first 2D matrix.
        mat2 (list of lists): The second 2D matrix.
        axis (int): The axis to concatenate along
                    (0 for rows, 1 for columns).

    Returns:
        list of lists: A new 2D matrix with the two matrices concatenated.
                       Returns None if they cannot be concatenated.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
