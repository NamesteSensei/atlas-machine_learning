#!/usr/bin/env python3
"""Defines a deep neural network for binary classification."""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network.

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

        prev_layer_size = nx
        for layer in range(1, self.__L + 1):
            self.__weights[f"W{layer}"] = (
                np.random.randn(layers[layer - 1], prev_layer_size)
                * np.sqrt(2 / prev_layer_size)
            )
            self.__weights[f"b{layer}"] = np.zeros((layers[layer - 1], 1))
            prev_layer_size = layers[layer - 1]

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """
        Perform forward propagation using a **single loop**.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            tuple: (final output A, updated cache dictionary).
        """
        self.__cache["A0"] = X
        output = X  # Track the activation output in one loop

        for layer in range(1, self.__L + 1):
            W = self.__weights[f"W{layer}"]
            b = self.__weights[f"b{layer}"]
            Z = np.matmul(W, output) + b
            output = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache[f"A{layer}"] = output

        return output, self.__cache
