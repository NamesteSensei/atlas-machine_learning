#!/usr/bin/env python3
"""
baum_welch module
Implements the Baum–Welch algorithm (EM) for Hidden Markov Models.
"""

import numpy as np

# Import dynamically since file names start with numbers
forward = __import__('3-forward').forward
backward = __import__('5-backward').backward


def baum_welch(Observations, Transition, Emission, Initial,
               iterations=1000):
    """
    Performs the Baum–Welch algorithm to train an HMM.

    Parameters:
        Observations (np.ndarray): array of shape (T,)
            indices of observations.
        Transition (np.ndarray): shape (M, M), transition probabilities.
        Emission (np.ndarray): shape (M, N), emission probabilities.
        Initial (np.ndarray): shape (M, 1), initial state probabilities.
        iterations (int): number of EM steps to perform.

    Returns:
        Transition (np.ndarray): updated transition matrix.
        Emission (np.ndarray): updated emission matrix.
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
        # Expectation Step
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

        # Maximization Step
        Transition = (np.sum(xi, axis=2)
                      / np.sum(gamma[:, :-1], axis=1).reshape(-1, 1))

        for k in range(N):
            mask = Observations == k
            Emission[:, k] = np.sum(gamma[:, mask], axis=1)
        Emission = Emission / np.sum(gamma, axis=1).reshape(-1, 1)

    return Transition, Emission
