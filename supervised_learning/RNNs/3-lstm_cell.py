#!/usr/bin/env python3
"""LSTM Cell module
Implements a single Long Short-Term Memory (LSTM) cell.
"""

import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""

    def __init__(self, i, h, o):
        """
        Initialize the LSTM cell.

        Args:
            i: Dimensionality of the data
            h: Dimensionality of the hidden state
            o: Dimensionality of the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Softmax activation"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """
        Perform forward propagation for one time step.

        Args:
            h_prev: previous hidden state, shape (m, h)
            c_prev: previous cell state, shape (m, h)
            x_t: input data at time t, shape (m, i)

        Returns:
            h_next, c_next, y
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f_t = self.sigmoid(
            np.matmul(concat, self.Wf) + self.bf
        )  # forget gate

        u_t = self.sigmoid(
            np.matmul(concat, self.Wu) + self.bu
        )  # update gate

        c_tilde = np.tanh(
            np.matmul(concat, self.Wc) + self.bc
        )  # candidate cell

        c_next = f_t * c_prev + u_t * c_tilde  # new cell state

        o_t = self.sigmoid(
            np.matmul(concat, self.Wo) + self.bo
        )  # output gate

        h_next = o_t * np.tanh(c_next)  # next hidden state

        y = self.softmax(
            np.matmul(h_next, self.Wy) + self.by
        )  # output vector

        return h_next, c_next, y
