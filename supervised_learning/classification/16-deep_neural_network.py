#!/usr/bin/env python3
"""Module: 16-deep_neural_network
Defines a deep neural network for binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.

    Private instance attributes:
        __L (int): Number of layers in the network.
        __cache (dict): Stores all intermediary values of the network.
        __weights (dict): Stores all weights and biases of the network.
            - Weights initialized using He et al. initialization.
    """

    def __init__(self, nx, layers):
        """
        Initialize a deep neural network.

        Args:
            nx (int): Number of input features.
            layers (list): List of positive integers representing the number
                of nodes in each layer.

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

        # ✅ Single loop for initializing weights & biases
        prev_layer_size = nx  # First layer takes input of size `nx`
        for i, nodes in enumerate(layers):
            self.__weights[f"W{i + 1}"] = (
                np.random.randn(nodes, prev_layer_size) *
                np.sqrt(2 / prev_layer_size)
            )
            self.__weights[f"b{i + 1}"] = np.zeros((nodes, 1))
            prev_layer_size = nodes  # Update for next iteration

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
