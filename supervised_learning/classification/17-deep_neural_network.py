#!/usr/bin/env python3
import numpy as np


class DeepNeuralNetwork:
    """
    Implements a deep neural network for binary classification.
    
    Attributes:
        L (int): Number of layers in the network.
        cache (dict): Stores activated outputs of each layer.
        weights (dict): Holds weights and biases of the network.
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network.

        Args:
            nx (int): Number of input features.
            layers (list): List of node counts per layer.

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

        prev_layer_size = nx  # Tracks size of the previous layer
        for layer in range(1, self.__L + 1):
            self.__weights[f"W{layer}"] = (
                np.random.randn(layers[layer - 1], prev_layer_size) *
                np.sqrt(2 / prev_layer_size)
            )
            self.__weights[f"b{layer}"] = np.zeros((layers[layer - 1], 1))
            prev_layer_size = layers[layer - 1]  # Update for next layer

    @property
    def L(self):
        """Returns number of layers."""
        return self.__L

    @property
    def cache(self):
        """Returns cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Returns weights dictionary."""
        return self.__weights

    def forward_prop(self, X):
        """
        Executes forward propagation.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            tuple: Activated output of last layer and cache.
        """
        self.__cache["A0"] = X  # Store input data
        prev_activation = X

        for layer in range(1, self.__L + 1):
            Z = np.matmul(self.__weights[f"W{layer}"], prev_activation) + self.__weights[f"b{layer}"]
            prev_activation = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache[f"A{layer}"] = prev_activation

        return prev_activation, self.__cache

