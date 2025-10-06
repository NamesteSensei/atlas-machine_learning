#!/usr/bin/env python3
"""
absorbing module
Determines if a Markov chain is absorbing.
"""

import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    Parameters:
        P (numpy.ndarray): Square 2D array (n, n), transition matrix.

    Returns:
        bool: True if absorbing, else False.
    """
    if (not isinstance(P, np.ndarray) or len(P.shape) != 2
            or P.shape[0] != P.shape[1]):
        return False

    n = P.shape[0]
    absorbing_states = np.where(np.isclose(np.diag(P), 1))[0]
    if len(absorbing_states) == 0:
        return False

    visited = np.zeros(n, dtype=bool)
    for state in absorbing_states:
        visited[state] = True

    for i in range(n):
        reachable = np.zeros(n, dtype=bool)
        queue = [i]
        while queue:
            current = queue.pop(0)
            for j in range(n):
                if P[current, j] > 0 and not reachable[j]:
                    reachable[j] = True
                    queue.append(j)
        if not any(reachable[absorbing_states]):
            return False
    return True
