#!/usr/bin/env python3
"""Module to initialize a Q-table for a FrozenLake environment."""

import numpy as np


def q_init(env):
    """
    Initialize a Q-table with zeros.

    Args:
        env (gym.Env): The FrozenLake environment instance.

    Returns:
        np.ndarray: A Q-table initialized to zeros with shape
                    (number of states, number of actions).
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    return np.zeros((num_states, num_actions))
