#!/usr/bin/env python3
"""Bidirectional RNN Forward Propagation"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    Args:
        bi_cell: instance of BidirectionalCell
        X: input data of shape (t, m, i)
        h_0: initial hidden state in forward direction, shape (m, h)
        h_t: initial hidden state in backward direction, shape (m, h)

    Returns:
        H: concatenated hidden states, shape (t, m, 2h)
        Y: outputs, shape (t, m, o)
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    # Initialize hidden state arrays
    H_f = np.zeros((t, m, h))
    H_b = np.zeros((t, m, h))

    # Forward pass
    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        H_f[step] = h_prev

    # Backward pass
    h_next = h_t
    for step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[step])
        H_b[step] = h_next

    # Concatenate forward and backward hidden states
    H = np.concatenate((H_f, H_b), axis=2)

    # Compute outputs using the bi_cell's output method
    Y = bi_cell.output(H)

    return H, Y
