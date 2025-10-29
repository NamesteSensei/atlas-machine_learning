#!/usr/bin/env python3
"""Deep RNN module
Performs forward propagation for a deep RNN.
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Perform forward propagation for a deep RNN.

    Args:
        rnn_cells: list of RNNCell instances of length num_layers
        X: input data, shape (t, m, i)
        h_0: initial hidden state, shape (num_layers, m, h)

    Returns:
        H: hidden states, shape (t + 1, num_layers, m, h)
        Y: outputs, shape (t, m, o)
    """
    t, m, _ = X.shape
    num_layers = len(rnn_cells)
    h = h_0.shape[2]
    o = rnn_cells[-1].by.shape[1]

    # Initialize hidden states and outputs
    H = np.zeros((t + 1, num_layers, m, h))
    H[0] = h_0
    Y = np.zeros((t, m, o))

    # Iterate through time steps
    for step in range(t):
        x_t = X[step]
        for layer in range(num_layers):
            h_prev = H[step, layer]
            rnn_cell = rnn_cells[layer]

            h_next, y = rnn_cell.forward(h_prev, x_t)
            H[step + 1, layer] = h_next

            # Output of current layer is input to next layer
            x_t = h_next

        # Final layer output stored in Y
        Y[step] = y

    return H, Y
