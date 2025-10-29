#!/usr/bin/env python3
"""RNN Cell module"""

import numpy as np


class RNNCell:
    """
    Represents a single cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
        - i (int): Dimensionality of the data (input size)
        - h (int): Dimensionality of the hidden state
        - o (int): Dimensionality of the output
        """
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step

        Parameters:
        - h_prev (np.ndarray): shape (m, h), previous hidden state
        - x_t (np.ndarray): shape (m, i), input at time t

        Returns:
        - h_next (np.ndarray): shape (m, h), next hidden state
        - y (np.ndarray): shape (m, o), output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)  # shape (m, i+h)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        y_raw = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y_raw)
        return h_next, y

    @staticmethod
    def softmax(x):
        """
        Apply softmax activation function

        Parameters:
        - x (np.ndarray): shape (m, o)

        Returns:
        - Softmax output (np.ndarray): same shape
        """
        x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))  # stability fix
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)
