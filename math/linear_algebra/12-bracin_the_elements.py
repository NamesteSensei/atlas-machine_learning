#!/usr/bin/env python3
"""
This module performs element-wise operations on two NumPy arrays.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division.

    Args:
        mat1 (numpy.ndarray): The first array.
        mat2 (numpy.ndarray or scalar): The second array or scalar.

    Returns:
        tuple: A tuple containing element-wise sum, difference, product, and quotient.
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
