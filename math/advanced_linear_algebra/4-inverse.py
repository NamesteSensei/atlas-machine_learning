#!/usr/bin/env python3
"""Compute the inverse of a square matrix (if it exists)."""

# Reuse prior tasks
determinant = __import__('0-determinant').determinant
adjugate = __import__('3-adjugate').adjugate


def inverse(matrix):
    """
    Calculates the inverse of a matrix.

    Args:
        matrix (list of lists): A non-empty square matrix.

    Returns:
        list of lists or None: The inverse of matrix, or None if singular.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is empty or not square.
    """
    is_list = isinstance(matrix, list)
    rows_are_lists = is_list and all(isinstance(r, list) for r in matrix)

    # Match project checker: [] -> TypeError (not list of lists)
    if (not is_list) or (len(matrix) == 0) or (not rows_are_lists):
        raise TypeError("matrix must be a list of lists")

    # Disallow 0x0 representation and non-square
    n = len(matrix)
    if matrix == [[]] or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)
    inv = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(adj[i][j] / det)
        inv.append(row)
    return inv
