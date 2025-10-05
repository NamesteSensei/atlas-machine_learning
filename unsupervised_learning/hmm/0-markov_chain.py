#!/usr/bin/env python3
"""
markov_chain module
Determines the probability of a Markov chain being in a particular
state after a specified number of iterations.
"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of being in a specific state after t iterations.

    Parameters:
    P (numpy.ndarray): Square 2D array of shape (n, n), transition matrix.
    s (numpy.ndarray): Row vector (1, n), initial state probabilities.
    t (int): Number of iterations.

    Returns:
    numpy.ndarray: (1, n) vector of probabilities after t steps, or None on failure.
    """
    if (not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray) or
            len(P.shape) != 2 or P.shape[0] != P.shape[1] or
            s.shape != (1, P.shape[0]) or not isinstance(t, int) or t < 0):
        return None

    # Raise matrix to power t
    P_t = np.linalg.matrix_power(P, t)

    # Compute final distribution
    result = np.matmul(s, P_t)

    return result
