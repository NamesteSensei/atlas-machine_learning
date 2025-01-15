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
    # Transpose the matrix using list comprehension
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
