#!/usr/bin/env python3
"""Deep Neural Network module"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network

        Parameters:
        nx (int): Number of input features
        layers (list): List representing the number of nodes in each layer

        Raises:
        TypeError: If nx is not an integer
        ValueError: If nx is less than 1
        TypeError: If layers is not a list of positive integers
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(nodes, int) and nodes > 0 for nodes in layers):
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.L = len(layers)  # Number of layers
        self.cache = {}       # Initialize cache dictionary
        self.weights = {}     # Initialize weights dictionary

        # He et al. initialization for weights and zeros for biases
        for l in range(1, self.L + 1):
            layer_size = layers[l - 1]
            if l == 1:
                prev_layer_size = nx
            else:
                prev_layer_size = layers[l - 2]

            # Weights initialization
            weight_key = f'W{l}'
            bias_key = f'b{l}'
            self.weights[weight_key] = (np.random.randn(layer_size, prev_layer_size) *
                                        np.sqrt(2 / prev_layer_size))
            # Biases initialization
            self.weights[bias_key] = np.zeros((layer_size, 1))
