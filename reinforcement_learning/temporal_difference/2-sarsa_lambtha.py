#!/usr/bin/env python3
"""SARSA(λ) algorithm for training a Q-table with eligibility traces"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1.0,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm with eligibility traces.

    Args:
        env: OpenAI Gym-like environment instance.
        Q: numpy.ndarray of shape (s, a) with the Q-table.
        lambtha: eligibility trace decay factor (0 ≤ λ ≤ 1).
        episodes: total number of episodes to train.
        max_steps: max steps per episode.
        alpha: learning rate.
        gamma: discount factor.
        epsilon: initial epsilon for epsilon-greedy.
        min_epsilon: minimum value for epsilon.
        epsilon_decay: decay rate for epsilon.

    Returns:
        Q: the updated Q-table.
    """
    n_states, n_actions = Q.shape

    for ep in range(episodes):
        state, _ = env.reset()
        elig = np.zeros((n_states, n_actions))

        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if np.random.rand() < epsilon:
                next_action = np.random.randint(n_actions)
            else:
                next_action = np.argmax(Q[next_state])

            td_target = reward
            td_target += gamma * Q[next_state, next_action] * (not done)
            td_error = td_target - Q[state, action]

            elig[state, action] += 1
            Q += alpha * td_error * elig
            elig *= gamma * lambtha

            if done:
                break

            state = next_state
            action = next_action

        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q
