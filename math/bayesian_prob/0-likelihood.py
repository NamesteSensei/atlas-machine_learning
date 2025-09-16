#!/usr/bin/env python3
"""
Module that calculates the likelihood of observing data
given hypothetical probabilities.
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculate the likelihood of seeing x patients with side
    effects out of n given probabilities in P.

    Args:
        x (int): number with side effects
        n (int): total number of patients
        P (np.ndarray): 1D array of probabilities

    Returns:
        np.ndarray: likelihood values for each prob in P

    Raises:
        ValueError: for invalid x, n, or probability ranges
        TypeError: if P is not a 1D numpy.ndarray
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is "
                         "greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the "
                         "range [0, 1]")

    fact = np.math.factorial
    comb = fact(n) / (fact(x) * fact(n - x))
    return comb * (P ** x) * ((1 - P) ** (n - x))
