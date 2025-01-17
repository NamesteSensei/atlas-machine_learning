#!/usr/bin/env python3
"""
Module to calculate the integral of a polynomial.
"""

def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    Args:
        poly (list): A list of coefficients where the index represents
                     the power of x for that coefficient.
        C (int): The constant of integration.

    Returns:
        list: The integral of the polynomial as a list of coefficients.
              Returns None if poly is not valid or if C is not an integer.
    """
    if not isinstance(poly, list) or not isinstance(C, (int, float)):
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    # Calculate integral: new coefficient = coeff / (power + 1)
    integral = [C] + [coeff / (power + 1) for power, coeff in enumerate(poly)]
    return [round(x, 2) if isinstance(x, float) else x for x in integral]
