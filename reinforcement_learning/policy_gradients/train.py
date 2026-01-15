#!/usr/bin/env python3
"""Training function for REINFORCE with Monte-Carlo policy gradients."""

import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Train a policy using the REINFORCE algorithm.

    Args:
        env: OpenAI gymnasium environment
        nb_episodes (int): number of episodes to train on
        alpha (float): learning rate
        gamma (float): discount factor

    Returns:
        list: total rewards per episode
    """
    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    weight = np.random.rand(n_features, n_actions)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads, rewards = [], []
        total_reward = 0

        done = False
        while not done:
            action, grad = policy_gradient(state, weight)
            state, reward, terminated, truncated, _ = env.step(action)
            grads.append(grad)
            rewards.append(reward)
            total_reward += reward
            done = terminated or truncated

        # Compute discounted returns
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)

        # Normalize returns (optional but improves stability)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Update weights using accumulated gradients
        for g, R in zip(grads, returns):
            weight += alpha * g * R

        print(f"Episode: {episode} Score: {total_reward}")
        scores.append(total_reward)

    return scores
