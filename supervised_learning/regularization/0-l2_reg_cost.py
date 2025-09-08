#!/usr/bin/env python3
"""
Compute L2-regularized cost in NumPy.
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Returns cost + (lambtha/(2m)) * sum(||W_l||^2) over layers.
    """
    l2_sum = 0.0
    for layer in range(1, L + 1):
        W = weights[f"W{layer}"]
        l2_sum += np.sum(W ** 2)
    return cost + (lambtha / (2 * m)) * l2_sum
