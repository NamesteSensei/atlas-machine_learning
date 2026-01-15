#!/usr/bin/env python3
"""Monte Carlo prediction algorithm using incremental updates."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Estimate the value function V using every-visit Monte Carlo prediction.

    Args:
        env: OpenAI Gym environment.
        V: np.ndarray, shape (n_states,), initial value estimates.
        policy: function mapping state -> action.
        episodes: int, number of episodes to run.
        max_steps: int, maximum steps per episode.
        alpha: float, learning rate.
        gamma: float, discount factor.

    Returns:
        Updated value estimates V.
    """
    for _ in range(episodes):
        state = env.reset()[0]
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            if terminated or truncated:
                break

        G = 0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            V[state] += alpha * (G - V[state])

    return V
