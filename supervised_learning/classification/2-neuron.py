#!/usr/bin/env python3
"""
Neuron Class for Binary Classification with Forward Propagation
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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
    @property
    def W(self):
        """Getter for W"""
        return self.__W
    @property
    def b(self):
        """Getter for b"""
        return self.__b
    @property
    def A(self):
        """Getter for A"""
        return self.__A
    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron.
        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
        Returns:
            numpy.ndarray: Activated output (A) of shape (1, m).
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
        return self.__A
