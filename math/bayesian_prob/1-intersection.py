#!/usr/bin/env python3
"""
Calculates the intersection (joint probability) of data and hypotheses.
"""

import numpy as np
from likelihood import likelihood


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of the data with the given probabilities.

    Parameters
    ----------
    x : int
        Number of patients with severe side effects
    n : int
        Total number of patients
    P : numpy.ndarray
        1D array of hypothetical probabilities
    Pr : numpy.ndarray
        1D array of prior beliefs (same shape as P)

    Returns
    -------
    numpy.ndarray
        Intersection values for each probability in P
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

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

    return likelihood(x, n, P) * Pr
