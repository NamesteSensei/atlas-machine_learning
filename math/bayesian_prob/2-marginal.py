#!/usr/bin/env python3
"""
Module that calculates the marginal probability of
observing data given priors and hypotheses.
"""
import numpy as np
intersection = __import__('1-intersection').intersection


def marginal(x, n, P, Pr):
    """
    Calculate the marginal probability of observing data.

    Args:
        x (int): patients with side effects
        n (int): total patients
        P (np.ndarray): hypothetical probabilities
        Pr (np.ndarray): prior beliefs

    Returns:
        float: marginal probability of the data

    Raises:
        ValueError, TypeError: according to spec
    """
    inter = intersection(x, n, P, Pr)
    return np.sum(inter)
