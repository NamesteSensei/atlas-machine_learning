#!/usr/bin/env python3
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.

    Private instance attributes:
        __L (int): The number of layers in the network.
        __cache (dict): A dictionary to hold all intermediary values.
        __weights (dict): A dictionary to hold all weights and biases.
            Weights are initialized using He et al. initialization.
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network.

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
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            layer_key_w = f"W{i + 1}"
            layer_key_b = f"b{i + 1}"
            if i == 0:
                self.__weights[layer_key_w] = (
                    np.random.randn(layers[i], nx) *
                    np.sqrt(2 / nx)
                )
            else:
                prev_layer = layers[i - 1]
                self.__weights[layer_key_w] = (
                    np.random.randn(layers[i], prev_layer) *
                    np.sqrt(2 / prev_layer)
                )
            self.__weights[layer_key_b] = np.zeros((layers[i], 1))

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
