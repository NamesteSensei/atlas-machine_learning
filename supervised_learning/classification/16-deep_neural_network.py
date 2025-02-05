#!/usr/bin/env python3
"""Defines a deep neural network for binary classification."""

import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.

    Private instance attributes:
        __L (int): The number of layers in the network.
        __cache (dict): Stores all intermediary values of the network.
        __weights (dict): Holds weights and biases, initialized using He et al.
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network.

        Args:
            nx (int): Number of input features.
            layers (list): List of positive integers representing the
                          number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If layers is not a list of positive integers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer_index in range(self.__L):
            weight_key = f"W{layer_index + 1}"
            bias_key = f"b{layer_index + 1}"

            if layer_index == 0:
                self.__weights[weight_key] = (
                    np.random.randn(layers[layer_index], nx) *
                    np.sqrt(2 / nx)
                )
            else:
                prev_nodes = layers[layer_index - 1]
                self.__weights[weight_key] = (
                    np.random.randn(layers[layer_index], prev_nodes) *
                    np.sqrt(2 / prev_nodes)
                )

            self.__weights[bias_key] = np.zeros((layers[layer_index], 1))

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
