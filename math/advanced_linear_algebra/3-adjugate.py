#!/usr/bin/env python3
"""Compute the adjugate (adjoint) of a square matrix."""

# Import cofactor() from 2-cofactor.py
cofactor = __import__('2-cofactor').cofactor


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a square matrix.

    Args:
        matrix (list of lists): A non-empty square matrix.

    Returns:
        list of lists: The adjugate (transpose of cofactor) of matrix.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is empty or not square.
    """
    cof = cofactor(matrix)  # validates types/shape and builds cofactors

    n = len(cof)
    adj = []
    for j in range(n):
        row = []
        for i in range(n):
            row.append(cof[i][j])
        adj.append(row)
    return adj
