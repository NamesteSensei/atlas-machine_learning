#!/usr/bin/env python3
"""
Module to calculate the derivative of a polynomial.
"""

def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial.

    Args:
        poly (list): A list of coefficients where the index represents
                     the power of x for that coefficient.

    Returns:
        list: The derivative of the polynomial as a list of coefficients.
              Returns [0] if the derivative is 0.
              Returns None if poly is not valid.
    """
    if not isinstance(poly, list) or not all(isinstance(c, (int, float)) for c in poly):
        return None
    if len(poly) == 1:
        return [0]
    
    return [coeff * power for power, coeff in enumerate(poly) if power > 0]
