#!/usr/bin/env python3
"""
baum_welch module
Implements the Baum–Welch algorithm (EM) for Hidden Markov Models.
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs the Forward algorithm."""
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


def backward(Observation, Emission, Transition, Initial):
    """Performs the Backward algorithm."""
    T = Observation.shape[0]
    N = Emission.shape[0]
    B = np.zeros((N, T))
    B[:, -1] = 1
    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(Transition[i] *
                             Emission[:, Observation[t + 1]] * B[:, t + 1])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B


def baum_welch(Observations, Transition, Emission, Initial,
               iterations=1000):
    """
    Performs the Baum–Welch algorithm to train an HMM.
    """
    if (not isinstance(Observations, np.ndarray)
            or not isinstance(Transition, np.ndarray)
            or not isinstance(Emission, np.ndarray)
            or not isinstance(Initial, np.ndarray)
            or len(Observations.shape) != 1
            or len(Transition.shape) != 2
            or Transition.shape[0] != Transition.shape[1]
            or Emission.shape[0] != Transition.shape[0]
            or Initial.shape[0] != Transition.shape[0]):
        return None, None

    T = Observations.shape[0]
    M, N = Emission.shape

    for _ in range(iterations):
        P, F = forward(Observations, Emission, Transition, Initial)
        _, B = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((M, M, T - 1))
        gamma = np.zeros((M, T))

        for t in range(T - 1):
            denom = np.dot(np.dot(F[:, t].T, Transition)
                           * Emission[:, Observations[t + 1]].T,
                           B[:, t + 1])
            for i in range(M):
                numer = (F[i, t] * Transition[i]
                         * Emission[:, Observations[t + 1]]
                         * B[:, t + 1])
                xi[i, :, t] = numer / denom
            gamma[:, t] = np.sum(xi[:, :, t], axis=1)

        gamma[:, T - 1] = F[:, T - 1] / np.sum(F[:, T - 1])

        Transition = (np.sum(xi, axis=2)
                      / np.sum(gamma[:, :-1], axis=1).reshape(-1, 1))

        for k in range(N):
            mask = Observations == k
            Emission[:, k] = np.sum(gamma[:, mask], axis=1)
        Emission = Emission / np.sum(gamma, axis=1).reshape(-1, 1)

    return Transition, Emission
