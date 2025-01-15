#!/usr/bin/env python3
"""
This module contains a function to transpose a 2D matrix.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.

    Args:
        matrix (list of lists): The 2D matrix to transpose.

    Returns:
        list of lists: A new 2D matrix that is the transpose of the input matrix.
    """
    return [
        [row[i] for row in matrix]  # E201 fixed: No extra space after '['
        for i in range(len(matrix[0]))
    ]
