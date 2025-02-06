#!/usr/bin/env python3
"""Module for DeepNeuralNetwork performing binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network for binary classification."""

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network.
        
        Args:
            nx (int): Number of input features.
            layers (list): List with the number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer or layers is not a list.
            ValueError: If nx is less than 1 or layers has non-positive ints.
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
        prev_layer = nx

        # ✅ SINGLE LOOP for weight initialization
        for l in range(1, self.__L + 1):
            self.__weights[f"W{l}"] = (
                np.random.randn(layers[l - 1], prev_layer) *
                np.sqrt(2 / prev_layer)
            )
            self.__weights[f"b{l}"] = np.zeros((layers[l - 1], 1))
            prev_layer = layers[l - 1]

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
            tuple: The final activated output and cache dictionary.
        """
        self.__cache["A0"] = X

        # ✅ SINGLE LOOP for forward propagation
        for l in range(1, self.__L + 1):
            W, b = self.__weights[f"W{l}"], self.__weights[f"b{l}"]
            Z = np.matmul(W, self.__cache[f"A{l - 1}"]) + b
            self.__cache[f"A{l}"] = 1 / (1 + np.exp(-Z))  # Sigmoid activation

        return self.__cache[f"A{self.__L}"], self.__cache
