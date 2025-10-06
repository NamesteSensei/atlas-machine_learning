#!/usr/bin/env python3
"""
forward module
Implements the Forward algorithm for a Hidden Markov Model (HMM).
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the Forward algorithm.

    Returns:
        P (float): likelihood of the observations.
        F (numpy.ndarray): forward probabilities.
    """
    if (not isinstance(Observation, np.ndarray)
            or len(Observation.shape) != 1):
        return None, None
    if (not isinstance(Emission, np.ndarray)
            or not isinstance(Transition, np.ndarray)
            or not isinstance(Initial, np.ndarray)):
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]
    F = np.zeros((N, T))
    F[:, 0] = (Initial.T * Emission[:, Observation[0]]).flatten()

    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[:, t - 1] * Transition[:, j]) * \
                Emission[j, Observation[t]]
    P = np.sum(F[:, -1])
    return P, F
