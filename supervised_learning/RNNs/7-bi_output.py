#!/usr/bin/env python3
"""Bidirectional RNN Cell - Output computation.
Computes all outputs for the bidirectional RNN.
"""

import numpy as np


class BidirectionalCell:
    """Represents a bidirectional RNN cell."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the BidirectionalCell.

        Args:
            input_dim: dimensionality of the data (i)
            hidden_dim: dimensionality of the hidden states (h)
            output_dim: dimensionality of the outputs (o)
        """
        # Forward direction
        self.Whf = np.random.randn(input_dim + hidden_dim, hidden_dim)
        self.bhf = np.zeros((1, hidden_dim))

        # Backward direction
        self.Whb = np.random.randn(input_dim + hidden_dim, hidden_dim)
        self.bhb = np.zeros((1, hidden_dim))

        # Output layer
        self.Wy = np.random.randn(2 * hidden_dim, output_dim)
        self.by = np.zeros((1, output_dim))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev: previous hidden state, shape (m, h)
            x_t: input data at time t, shape (m, i)

        Returns:
            h_next: next hidden state in forward direction
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Perform backward propagation for one time step.

        Args:
            h_next: next hidden state, shape (m, h)
            x_t: input data at time t, shape (m, i)

        Returns:
            h_prev: previous hidden state in backward direction
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(concat, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """
        Calculate all outputs for the RNN.

        Args:
            H: concatenated hidden states, shape (t, m, 2 * h)

        Returns:
            Y: outputs, shape (t, m, o)
        """
        t, m, _ = H.shape
        y_linear = np.matmul(H, self.Wy) + self.by

        # Apply softmax along the output dimension
        exp_scores = np.exp(y_linear - np.max(y_linear, axis=2, keepdims=True))
        Y = exp_scores / np.sum(exp_scores, axis=2, keepdims=True)
        return Y
