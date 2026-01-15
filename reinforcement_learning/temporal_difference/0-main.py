#!/usr/bin/env python3
"""
Main test file for monte_carlo function with FrozenLake8x8-v1 environment.
"""

import gymnasium as gym
import numpy as np
import random
monte_carlo = __import__('0-monte_carlo').monte_carlo


def set_seed(env, seed=0):
    """
    Set reproducible seed for environment and random generators.
    """
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)


# Initialize FrozenLake environment
env = gym.make('FrozenLake8x8-v1')
set_seed(env, 0)

# Define action mappings
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


def policy(state):
    """
    A simple stochastic policy avoiding holes (H).
    Chooses between RIGHT, DOWN, UP, LEFT in that order of preference,
    depending on boundaries and holes.
    """
    p = np.random.uniform()
    row, col = state // 8, state % 8
    desc = env.unwrapped.desc

    if p > 0.5:
        if col != 7 and desc[row, col + 1] != b'H':
            return RIGHT
        elif row != 7 and desc[row + 1, col] != b'H':
            return DOWN
        elif row != 0 and desc[row - 1, col] != b'H':
            return UP
        else:
            return LEFT
    else:
        if row != 7 and desc[row + 1, col] != b'H':
            return DOWN
        elif col != 7 and desc[row, col + 1] != b'H':
            return RIGHT
        elif col != 0 and desc[row, col - 1] != b'H':
            return LEFT
        else:
            return UP


# Initialize value function: -1 for holes, +1 for others
V = np.where(env.unwrapped.desc == b'H', -1, 1).reshape(64).astype('float64')

# Display float precision
np.set_printoptions(precision=4)

# Run Monte Carlo and print value function as 8x8 grid
print(monte_carlo(env, V, policy).reshape((8, 8)))
