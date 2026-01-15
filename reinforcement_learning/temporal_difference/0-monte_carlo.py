#!/usr/bin/env python3
"""Monte Carlo prediction algorithm."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs Monte Carlo prediction.

    Args:
        env: Gymnasium environment
        V: np.ndarray of shape (s,) with value estimates
        policy: function mapping state to action
        episodes: number of episodes to run
        max_steps: max steps per episode
        alpha: learning rate
        gamma: discount factor

    Returns:
        Updated value table V
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
