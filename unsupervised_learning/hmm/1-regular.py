#!/usr/bin/env python3
"""
regular module
Determines the steady state probabilities of a regular Markov chain.
"""

import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular Markov chain.

    Parameters:
        P (numpy.ndarray): Square 2D array (n, n), transition matrix.

    Returns:
        numpy.ndarray: (1, n) steady-state probabilities, or None on
        failure.
    """
    if (not isinstance(P, np.ndarray) or len(P.shape) != 2
            or P.shape[0] != P.shape[1]):
        return None

    n = P.shape[0]
    P_power = np.linalg.matrix_power(P, 100)

    if np.any(P_power <= 0):
        return None

    eigvals, eigvecs = np.linalg.eig(P.T)
    steady_idx = np.argmin(np.abs(eigvals - 1))
    steady_state = eigvecs[:, steady_idx].real
    steady_state = steady_state / np.sum(steady_state)

    return steady_state.reshape(1, n)
