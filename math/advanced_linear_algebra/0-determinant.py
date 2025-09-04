#!/usr/bin/env python3
"""
0-determinant.py - Computes the determinant of a square matrix
"""

from typing import List


def determinant(matrix: List[List[float]]) -> float:
    """
    Computes the determinant of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix.

    Returns:
        The determinant as a float or int.

    Raises:
        TypeError: If input is not a list of lists.
        ValueError: If the matrix is not square.
    """
    # Check for list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Handle empty 0x0 matrix [[]]
    if matrix == [[]]:
        return 1

    # Check square shape
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case: 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case: 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case: expand along first row
    det = 0
    for j in range(n):
        # Create submatrix excluding row 0 and column j
        sub = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub)
    return det
