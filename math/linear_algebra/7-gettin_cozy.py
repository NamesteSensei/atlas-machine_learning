#!/usr/bin/env python3
"""
This module contains a function to concatenate two 2D matrices along a specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Args:
        mat1 (list of lists): First 2D matrix.
        mat2 (list of lists): Second 2D matrix.
        axis (int, optional): Axis to concatenate along. Defaults to 0.

    Returns:
        list of lists or None: A new matrix or None if shapes are incompatible.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None
