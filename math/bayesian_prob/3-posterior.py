#!/usr/bin/env python3
"""
Module to calculate the posterior probability distribution
over hypothetical probabilities P using Bayes' Theorem.

This is a self-contained version. Uses numpy only.
"""

import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability of each hypothetical
    probability in P given the observed data (x, n) and prior Pr.

    Uses Bayes’ theorem:
        posterior = (likelihood * prior) / marginal

    Parameters:
    ----------
    x : int
        Number of patients with severe side effects
    n : int
        Total number of patients observed
    P : numpy.ndarray
        1D array of hypothetical probabilities
    Pr : numpy.ndarray
        1D array of prior beliefs (same shape as P)

    Returns:
    -------
    numpy.ndarray
        1D array of posterior probabilities

    Raises:
    ------
    ValueError: For bad values of x, n, P, or Pr
    TypeError: If P or Pr are not numpy arrays of correct shape
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    def factorial(k):
        """Compute factorial iteratively."""
        if k == 0 or k == 1:
            return 1
        result = 1
        for i in range(2, k + 1):
            result *= i
        return result

    def binom_coeff(n, k):
        """Compute binomial coefficient (n choose k)."""
        return factorial(n) / (factorial(k) * factorial(n - k))

    # Likelihood * Prior = Joint
    likelihood = binom_coeff(n, x) * (P ** x) * ((1 - P) ** (n - x))
    joint = likelihood * Pr

    # Marginal = sum of joint
    marginal = np.sum(joint)

    # Posterior = joint / marginal
    return joint / marginal
