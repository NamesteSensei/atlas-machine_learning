#!/usr/bin/env python3
"""Policy gradient function implementation using softmax."""

import numpy as np


def policy(state, weight):
    """
    Compute the action probabilities (policy) using a softmax function.

    Args:
        state (np.ndarray): shape (1, n_features), the current state.
        weight (np.ndarray): shape (n_features, n_actions), policy weights.

    Returns:
        np.ndarray: shape (1, n_actions), probabilities of each action.
    """
    z = state @ weight            # Linear combination (logits)
    z -= np.max(z)                # For numerical stability in softmax
    exp = np.exp(z)               # Exponentiate
    return exp / np.sum(exp, axis=1, keepdims=True)  # Softmax normalize
