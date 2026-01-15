#!/usr/bin/env python3
"""
Monte Carlo prediction algorithm for estimating the value function V
under a policy.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm.

    Args:
        env: The OpenAI Gymnasium environment instance.
        V: A NumPy array of shape (s,) containing the value estimates.
        policy: A function that maps state to action.
        episodes: Total number of episodes to train.
        max_steps: Max steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.

    Returns:
        Updated value function V.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            if done or truncated:
                break

        G = 0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            V[state] += alpha * (G - V[state])

    return V
