#!/usr/bin/env python3
"""
Task 9: Calculate the sum of squares up to a given number `n` without loops.

This script uses a mathematical formula to compute the sum of squares from
1 to `n`.
"""


def summation_i_squared(n):
    """
    Calculate the sum of squares from 1 to `n` using a mathematical formula.

    Args:
        n (int): The stopping condition (upper limit of the sum).

    Returns:
        int: The sum of squares up to `n`, or None if `n` is invalid.
    """
    if not isinstance(n, int) or n <= 0:
        return None
    # Using the formula for the sum of squares: n(n+1)(2n+1)/6
    return n * (n + 1) * (2 * n + 1) // 6
