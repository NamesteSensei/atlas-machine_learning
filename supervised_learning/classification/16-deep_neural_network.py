#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""

import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network

        Parameters:
        nx (int): Number of input features
        layers (list): Number of nodes in each layer

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

        # Initialize attributes
        self.nx = nx
        self.layers = layers
        self.L = len(layers)  # Number of layers
        self.cache = {}       # Dictionary to hold intermediary values
        self.weights = {}     # Dictionary to hold weights and biases

        # He et al. initialization for weights and biases
        for layer_idx in range(1, self.L + 1):
            layer_key_W = 'W{}'.format(layer_idx)
            layer_key_b = 'b{}'.format(layer_idx)

            if layer_idx == 1:
                weight_shape = (layers[layer_idx - 1], nx)
                weight_init = np.random.randn(*weight_shape) * np.sqrt(2 / nx)
            else:
                weight_shape = (layers[layer_idx - 1], layers[layer_idx - 2])
                prev_nodes = layers[layer_idx - 2]
                weight_init = (np.random.randn(*weight_shape) *
                               np.sqrt(2 / prev_nodes))

            self.weights[layer_key_W] = weight_init
            self.weights[layer_key_b] = np.zeros((layers[layer_idx - 1], 1))
