#!/usr/bin/env python3
"""
backward module
Performs the Backward algorithm for a Hidden Markov Model (HMM).
"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the Backward algorithm.

    Returns:
        P (float): likelihood of the observations.
        B (numpy.ndarray): backward probabilities.
    """
    T = Observation.shape[0]
    N = Emission.shape[0]
    B = np.zeros((N, T))
    B[:, -1] = 1

    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(Transition[i] * Emission[:, Observation[t + 1]]
                             * B[:, t + 1])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B
