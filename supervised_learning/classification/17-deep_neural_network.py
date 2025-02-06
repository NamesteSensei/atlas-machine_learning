#!/usr/bin/env python3
"""
Defines a deep neural network performing forward propagation.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Class that defines a deep neural network for binary classification.
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network.

        Args:
            nx (int): Number of input features.
            layers (list): Number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If layers is not a list of positive integers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if (not isinstance(layers, list) or len(layers) == 0 or
                not all(isinstance(n, int) and n > 0 for n in layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev_layer_size = nx  # Track previous layer size
        for layer_index, nodes in enumerate(layers, start=1):
            self.__weights[f"W{layer_index}"] = (
                np.random.randn(nodes, prev_layer_size) *
                np.sqrt(2 / prev_layer_size)
            )
            self.__weights[f"b{layer_index}"] = np.zeros((nodes, 1))
            prev_layer_size = nodes  # Update for next layer

    @property
    def L(self):
        """Getter for number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """
        Performs forward propagation using sigmoid activation.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            tuple: (A, cache) where:
                - A is the final layer activation.
                - cache contains all intermediate values.
        """
        self.__cache["A0"] = X  # Store input layer
        for layer in range(1, self.__L + 1):
            W = self.__weights[f"W{layer}"]
            b = self.__weights[f"b{layer}"]
            Z = np.matmul(W, self.__cache[f"A{layer - 1}"]) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache[f"A{layer}"] = A

        return A, self.__cache
