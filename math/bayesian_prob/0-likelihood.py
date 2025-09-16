#!/usr/bin/env python3
"""
Calculates the likelihood of observing x successes in n trials
for each hypothetical probability in probabilities.
"""

import numpy as np


def likelihood(x, n, probabilities):
    """
    Parameters:
    - x: number of patients with severe side effects
    - n: total number of patients
    - probabilities: 1D numpy.ndarray of hypothetical probabilities

    Returns:
    - 1D numpy.ndarray of likelihood values
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than "
                         "or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(probabilities, np.ndarray) or probabilities.ndim != 1:
        raise TypeError("probabilities must be a 1D numpy.ndarray")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("All values in probabilities must be in the "
                         "range [0, 1]")

    coefficient = np.math.comb(n, x)
    return coefficient * (probabilities ** x) * (
        (1 - probabilities) ** (n - x)
    )
