#!/usr/bin/env python3
"""
Main test file for monte_carlo function with FrozenLake8x8-v1 environment.
"""

import gymnasium as gym
import numpy as np
monte_carlo = __import__('0-monte_carlo').monte_carlo


def set_seed(env, seed=1):
    """
    Set reproducible seed for environment and numpy RNG.
    """
    env.reset(seed=seed)
    np.random.seed(seed)


# Initialize FrozenLake environment
env = gym.make('FrozenLake8x8-v1')
set_seed(env, 1)

# Action mappings
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3


def policy(state):
    """
    Fixed policy always taking the > 0.5 path (RIGHT, DOWN, UP, LEFT).
    """
    p = 1.0  # Force > 0.5
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
        return UP  # not used


# Initialize value table: -1 for holes, 1 for others
V = np.where(env.unwrapped.desc == b'H', -1, 1).reshape(64).astype('float64')

# Format output
np.set_printoptions(precision=4, suppress=True)

# Run Monte Carlo with exact params
print(monte_carlo(env, V, policy, episodes=50000, alpha=0.1, gamma=0.99).reshape((8, 8)))
