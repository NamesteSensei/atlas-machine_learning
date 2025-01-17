#!/usr/bin/env python3

def summation_i_squared(n):
    """
    Calculate the sum of squares from 1 to `n` using a mathematical formula.

    Args:
        n (int): The stopping condition of the summation.

    Returns:
        int: The calculated sum of squares.
        None: If `n` is not a valid positive integer.
    """
    if not isinstance(n, int) or n <= 0:
        return None

    # This uses a mathematical formula: n(n+1)(2n+1)/6
    return n * (n + 1) * (2 * n + 1) // 6
