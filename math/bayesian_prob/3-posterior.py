#!/usr/bin/env python3
"""
Module that calculates the posterior probability of
hypothetical probabilities given observed data.
"""
import numpy as np
intersection = __import__('1-intersection').intersection
marginal = __import__('2-marginal').marginal


def posterior(x, n, P, Pr):
    """
    Calculate the posterior probability for each hypothesis.

    Args:
        x (int): patients with side effects
        n (int): total patients
        P (np.ndarray): hypothetical probabilities
        Pr (np.ndarray): prior beliefs

    Returns:
        np.ndarray: posterior probabilities

    Raises:
        ValueError, TypeError: according to spec
    """
    inter = intersection(x, n, P, Pr)
    marg = marginal(x, n, P, Pr)
    return inter / marg
