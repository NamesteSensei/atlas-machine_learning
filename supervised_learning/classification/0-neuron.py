#!/usr/bin/env python3
"""
Neuron Class for Binary Classification
"""
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Initialize a Neuron.

        Args:
            nx (int): Number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0


# Test script
if __name__ == "__main__":
    np.random.seed(0)
    nx = 784
    neuron = Neuron(nx)
    print(neuron.W)
    print(neuron.W.shape)
    print(neuron.b)
    print(neuron.A)
    neuron.A = 10
    print(neuron.A)
