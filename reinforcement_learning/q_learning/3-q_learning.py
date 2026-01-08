#!/usr/bin/env python3
"""Q-learning algorithm for training an agent on FrozenLake."""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Train an agent using the Q-learning algorithm.

    Args:
        env (gym.Env): The environment to train on.
        Q (np.ndarray): The Q-table to update.
        episodes (int): Number of training episodes.
        max_steps (int): Max steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Initial exploration rate.
        min_epsilon (float): Minimum epsilon after decay.
        epsilon_decay (float): Epsilon decay rate per episode.

    Returns:
        tuple: (Q, total_rewards)
            Q: The updated Q-table.
            total_rewards: List of total rewards per episode.
    """
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminated and reward == 0:
                reward = -1

            # Q-learning update
            best_next_action = np.max(Q[next_state])
            td_target = reward + gamma * best_next_action
            td_delta = td_target - Q[state, action]
            Q[state, action] += alpha * td_delta

            state = next_state
            episode_reward += reward

            if done:
                break

        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay))
        total_rewards.append(episode_reward)

    return Q, total_rewards
