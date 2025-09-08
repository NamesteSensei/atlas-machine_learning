#!/usr/bin/env python3
"""Compute the cofactor matrix of a square matrix."""

# Import minor() from 1-minor.py
minor = __import__('1-minor').minor


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a square matrix.

    Args:
        matrix (list of lists): A non-empty square matrix.

    Returns:
        list of lists: The cofactor matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is empty or not square.
    """
    # Let minor() perform all input validation (TypeError / ValueError).
    minors = minor(matrix)

    n = len(minors)
    cofs = []
    for i in range(n):
        row = []
        for j in range(n):
            sign = -1 if ((i + j) % 2) else 1
            row.append(sign * minors[i][j])
        cofs.append(row)
    return cofs
