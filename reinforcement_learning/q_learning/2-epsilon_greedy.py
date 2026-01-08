#!/usr/bin/env python3
"""Module for epsilon-greedy action selection."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Use epsilon-greedy to choose the next action.

    Args:
        Q (np.ndarray): Q-table.
        state (int): Current state.
        epsilon (float): Exploration rate (0 ≤ ε ≤ 1).

    Returns:
        int: Index of the action to take.
    """
    if np.random.uniform(0., 1.) < epsilon:
        # Explore: pick a random action
        return np.random.randint(Q.shape[1])
    else:
        # Exploit: pick the action with the highest Q-value
        return np.argmax(Q[state])
