#!/usr/bin/env python3
"""Play one episode using a trained Q-table."""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Play an episode using the trained Q-table.

    Args:
        env (gym.Env): FrozenLake environment.
        Q (np.ndarray): Trained Q-table.
        max_steps (int): Max number of steps in episode.

    Returns:
        tuple: (total_reward, rendered_outputs)
            total_reward (float): Sum of rewards collected.
            rendered_outputs (list): List of strings rendering the board.
    """
    state, _ = env.reset()
    total_reward = 0
    rendered_outputs = [env.render()]

    for _ in range(max_steps):
        action = np.argmax(Q[state])  # Always exploit
        next_state, reward, terminated, truncated, _ = env.step(action)
        rendered_outputs.append(env.render())

        total_reward += reward
        state = next_state

        if terminated or truncated:
            break

    return total_reward, rendered_outputs
