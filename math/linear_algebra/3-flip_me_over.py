#!/usr/bin/env python3
"""
This module contains a function to transpose a 2D matr.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.
    Args:
        matrix (list of lists): The 2D matrix to tranpose.
    Returns:
        list of lists: A new 2D matrix is the tanspose of the input matrix.
    """
    return [
        [row[i] for row in matrix]
        for i in range(len(matrix[0]))
    ]
