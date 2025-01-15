#!/usr/bin/env python3
def matrix_shape(matrix):
    """Calculate the shape of a matrix."""
    if not isinstance(matrix, list):
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
