#!/usr/bin/env python3
"""TD(λ) algorithm for estimating value function"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm.

    Args:
        env: OpenAI gym-like environment instance.
        V: numpy.ndarray of shape (s,) containing value estimates.
        policy: function that returns action given a state.
        lambtha: eligibility trace decay parameter (0 ≤ λ ≤ 1).
        episodes: number of episodes to train on.
        max_steps: max steps per episode.
        alpha: learning rate (0 < α ≤ 1).
        gamma: discount factor (0 < γ ≤ 1).

    Returns:
        V: updated value function (numpy.ndarray).
    """
    n_states = V.shape[0]

    for ep in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros(n_states)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TD Error (δ)
            td_error = reward + gamma * V[next_state] * (not done) - V[state]

            # Update eligibility trace
            eligibility[state] += 1

            # Update value estimates and decay eligibility
            V += alpha * td_error * eligibility
            eligibility *= gamma * lambtha

            if done:
                break
            state = next_state

    return V
