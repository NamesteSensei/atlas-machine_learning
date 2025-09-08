#!/usr/bin/env python3
"""Calculate the determinant of a square matrix."""


def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    Args:
        matrix (list of lists): A square matrix.

    Returns:
        int or float: Determinant of the matrix.

    Raises:
        TypeError: If input is not a list of lists.
        ValueError: If matrix is not square.
    """
    is_list = isinstance(matrix, list)
    rows_are_lists = is_list and all(isinstance(r, list) for r in matrix)

    if (not is_list) or (len(matrix) == 0) or (not rows_are_lists):
        raise TypeError("matrix must be a list of lists")

    # The empty 0x0 matrix is represented as [[]]
    if matrix == [[]]:
        return 1

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        sub = [row[:col] + row[col + 1:] for row in matrix[1:]]
        sign = -1 if (col % 2) else 1
        det += sign * matrix[0][col] * determinant(sub)
    return det
