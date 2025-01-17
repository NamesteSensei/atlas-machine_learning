#!/usr/bin/env python3

def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.
    
    Args:
        poly: A list of coefficients representing a polynomial.
        C: An integer representing the integration constant.
    
    Returns:
        A new list of coefficients representing the integral of the polynomial.
        If poly or C are not valid, returns None.
    """
    if not isinstance(poly, list) or not all(isinstance(x, (int, float)) for x in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    integral = [C]
    for i, coef in enumerate(poly):
        integral.append(coef / (i + 1))
    
    # Remove trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    
    return [int(x) if isinstance(x, float) and x.is_integer() else x for x in integral]

# Example usage
if __name__ == "__main__":
    poly = [5, 3, 0, 1]
    print(poly_integral(poly))
