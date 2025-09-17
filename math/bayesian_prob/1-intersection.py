#!/usr/bin/env python3
"""
Module to compute the intersection (likelihood) of observing
x patients with side effects out of n total patients for each
hypothetical probability in P.

Implements binomial likelihood manually using numpy only.
"""

import numpy as np


def intersection(x, n, P):
    """
    Calculates the likelihood of observing `x` severe side effects
    out of `n` patients for each hypothetical probability in P.

    Parameters:
    ----------
    x : int
        Number of patients with severe side effects
    n : int
        Total number of patients observed
    P : numpy.ndarray
        1D array of hypothetical probabilities of side effects

    Returns:
    -------
    numpy.ndarray
        1D array of likelihood values corresponding to each P

    Raises:
    ------
    ValueError: For invalid n, x, or values in P
    TypeError: If P is not a 1D numpy.ndarray
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    def factorial(k):
        """Iterative factorial"""
        if k == 0 or k == 1:
            return 1
        result = 1
        for i in range(2, k + 1):
            result *= i
        return result

    def binom_coeff(n, k):
        """Binomial coefficient n choose k"""
        return factorial(n) / (factorial(k) * factorial(n - k))

    likelihood = binom_coeff(n, x) * (P ** x) * ((1 - P) ** (n - x))
    return likelihood
