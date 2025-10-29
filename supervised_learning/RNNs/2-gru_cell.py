#!/usr/bin/env python3
"""
Defines the GRUCell class for one step of a Gated Recurrent Unit.
"""
import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit (GRU) cell.
    """

    def __init__(self, i, h, o):
        """
        Class constructor.

        Parameters
        ----------
        i : int
            Dimensionality of the data.
        h : int
            Dimensionality of the hidden state.
        o : int
            Dimensionality of the outputs.
        """
        # Update gate parameters
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))

        # Reset gate parameters
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))

        # Candidate hidden state parameters
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))

        # Output layer parameters
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step of the GRU.

        Parameters
        ----------
        h_prev : np.ndarray of shape (m, h)
            Previous hidden state.
        x_t : np.ndarray of shape (m, i)
            Data input for the current time step.

        Returns
        -------
        h_next : np.ndarray of shape (m, h)
            The next hidden state.
        y : np.ndarray of shape (m, o)
            The output of the cell.
        """
        # Concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate (z_t)
        z_t = self.sigmoid(np.dot(concat, self.Wz) + self.bz)

        # Reset gate (r_t)
        r_t = self.sigmoid(np.dot(concat, self.Wr) + self.br)

        # Candidate hidden state (h̃_t)
        concat_reset = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.dot(concat_reset, self.Wh) + self.bh)

        # Final hidden state (h_next)
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Output (y_t)
        y_linear = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        return h_next, y
