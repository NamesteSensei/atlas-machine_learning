#!/usr/bin/env python3
"""
Calculates the intersection (joint probability) of data and hypotheses.
"""

import numpy as np
likelihood = __import__('0-likelihood').likelihood


def intersection(x, n, probabilities, priors):
    """
    Returns the joint probability of seeing x out of n
    for each value in probabilities, weighted by priors.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(probabilities, np.ndarray) or probabilities.ndim != 1:
        raise TypeError("probabilities must be a 1D numpy.ndarray")

    if (not isinstance(priors, np.ndarray) or
            priors.shape != probabilities.shape):
        raise TypeError(
            "priors must be a numpy.ndarray with the same shape "
            "as probabilities"
        )

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(
            "All values in probabilities must be in the range [0, 1]"
        )

    if np.any((priors < 0) | (priors > 1)):
        raise ValueError(
            "All values in priors must be in the range [0, 1]"
        )

    if not np.isclose(np.sum(priors), 1):
        raise ValueError("priors must sum to 1")

    return likelihood(x, n, probabilities) * priors
