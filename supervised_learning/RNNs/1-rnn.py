#!/usr/bin/env python3
"""
Performs forward propagation for a simple RNN
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Parameters
    ----------
    rnn_cell : RNNCell
        Instance of RNNCell used for forward propagation.
    X : np.ndarray of shape (t, m, i)
        Data to be used.
    h_0 : np.ndarray of shape (m, h)
        Initial hidden state.

    Returns
    -------
    H : np.ndarray of shape (t + 1, m, h)
        Array containing all hidden states.
    Y : np.ndarray of shape (t, m, o)
        Array containing all outputs.
    """
    t, m, _ = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
