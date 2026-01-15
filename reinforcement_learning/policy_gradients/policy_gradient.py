#!/usr/bin/env python3
"""Policy gradient functions for REINFORCE algorithm."""

import numpy as np


def policy(state, weight):
    """
    Compute the policy (action probabilities) using softmax.

    Args:
        state (np.ndarray): shape (1, n_features)
        weight (np.ndarray): shape (n_features, n_actions)

    Returns:
        np.ndarray: shape (1, n_actions)
    """
    z = state @ weight
    z -= np.max(z)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Compute the Monte-Carlo policy gradient.

    Args:
        state (np.ndarray): current environment state
        weight (np.ndarray): policy weight matrix

    Returns:
        tuple: (action, gradient)
    """
    state = state.reshape(1, -1)

    # Compute action probabilities
    probs = policy(state, weight)[0]

    # Sample action
    action = np.random.choice(len(probs), p=probs)

    # One-hot encode the action
    action_vec = np.zeros_like(probs)
    action_vec[action] = 1

    # Compute gradient
    grad = state.T @ (action_vec - probs).reshape(1, -1)

    return action, grad
