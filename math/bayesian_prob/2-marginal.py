#!/usr/bin/env python3
"""
Module that calculates the marginal probability of observing
x patients with severe side effects out of n total patients,
based on a prior distribution over hypothetical probabilities.

Only uses numpy, and implements binomial coefficient manually.
"""

import numpy as np


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of observing `x` patients
    with severe side effects out of `n` total patients, given
    a prior distribution over possible probabilities in `P`.

    Marginal probability is computed using the law of total probability:
        marginal = sum(Pr[i] * P(X=x | P[i])) for all i

    Parameters:
    ----------
    x : int
        Number of patients with severe side effects
    n : int
        Total number of patients observed
    P : numpy.ndarray
        1D array of hypothetical probabilities of side effects
    Pr : numpy.ndarray
        1D array of prior beliefs (same shape as P), must sum to 1

    Returns:
    -------
    float
        Marginal probability of observing the given data

    Raises:
    ------
    ValueError: If any of the following are violated:
        - n is not a positive integer
        - x is not a non-negative integer
        - x > n
        - values in P or Pr not in [0, 1]
        - Pr does not sum to 1
    TypeError: If P is not a 1D numpy.ndarray
        or Pr does not have the same shape as P
    """
    # === Input validation (strict order) ===
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
        """Compute factorial of k (k!) using iterative approach."""
        if k == 0 or k == 1:
            return 1
        result = 1
        for i in range(2, k + 1):
            result *= i
        return result

    def binom_coeff(n, k):
        """Compute binomial coefficient (n choose k)."""
        return factorial(n) / (factorial(k) * factorial(n - k))

    likelihood = binom_coeff(n, x) * (P ** x) * ((1 - P) ** (n - x))
    marginal_prob = np.sum(likelihood * Pr)

    return marginal_prob
