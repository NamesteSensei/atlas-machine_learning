#!/usr/bin/env python3
def summation_i_squared(n):
    """
    Calculate the sum of squares from 1 to n.

    Args:
        n (int): The stopping condition.

    Returns:
        int: The sum of squares from 1 to n, or None if n is invalid.
    """
    if not isinstance(n, int) or n <= 0:
        return None
    # Using the formula for the sum of squares: n(n+1)(2n+1)/6
    return n * (n + 1) * (2 * n + 1) // 6
