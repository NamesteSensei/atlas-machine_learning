#!/usr/bin/env python3
"""Compute the minor matrix of a square matrix."""

# Import determinant from file named "0-determinant.py"
determinant = __import__('0-determinant').determinant


def minor(matrix):
    """
    Calculates the minor matrix of a square matrix.

    Args:
        matrix (list of lists): A non-empty square matrix.

    Returns:
        list of lists: The minor matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is empty or not square.
    """
    is_list = isinstance(matrix, list)
    rows_are_lists = is_list and all(isinstance(r, list) for r in matrix)

    # Match the checker behavior: [] -> TypeError (list of lists)
    if (not is_list) or (len(matrix) == 0) or (not rows_are_lists):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # For 1x1 matrices, the minor matrix is [[1]]
    if n == 1:
        return [[1]]

    minors = []
    for i in range(n):
        row_minors = []
        for j in range(n):
            sub = [r[:j] + r[j + 1:] for k, r in enumerate(matrix) if k != i]
            row_minors.append(determinant(sub))
        minors.append(row_minors)
    return minors
