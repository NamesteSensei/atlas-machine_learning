#!/usr/bin/env python3
"""
Module to calculate the posterior probability distribution
over hypothetical probabilities P using Bayes' Theorem.

Reuses intersection() and marginal() from previous modules.
"""

import numpy as np
intersection = __import__('1-intersection').intersection
marginal = __import__('2-marginal').marginal


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

    # Calculate intersection and marginal
    joint_probs = intersection(x, n, P, Pr)
    marginal_prob = marginal(x, n, P, Pr)

    # Bayes' Theorem
    posteriors = joint_probs / marginal_prob
    return posteriors
