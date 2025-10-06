#!/usr/bin/env python3
"""
viterbi module
Calculates the most likely sequence of hidden states for an HMM.
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates most likely state sequence using the Viterbi algorithm.
    """
    T = Observation.shape[0]
    N = Transition.shape[0]
    V = np.zeros((N, T))
    B = np.zeros((N, T), dtype=int)

    V[:, 0] = (Initial.T * Emission[:, Observation[0]]).flatten()

    for t in range(1, T):
        for j in range(N):
            temp = V[:, t - 1] * Transition[:, j] * \
                Emission[j, Observation[t]]
            B[j, t] = np.argmax(V[:, t - 1] * Transition[:, j])
            V[j, t] = np.max(temp)

    P = np.max(V[:, -1])
    path = [np.argmax(V[:, -1])]
    for t in range(T - 1, 0, -1):
        path.insert(0, B[path[0], t])

    return path, P
