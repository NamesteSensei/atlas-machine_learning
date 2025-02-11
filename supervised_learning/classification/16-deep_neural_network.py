#!/usr/bin/env python3
import numpy as np

"""
/**
   This module implements a deep neural network used in binary classification.
   It sets up the count of layers, a storage of intermediate values, and the
   weights along with zero biases. The weights are set using the He method.
*/
"""

class DeepNeuralNetwork:
    """
    /**
       Deep neural network class used in binary classification.
       Public attributes:
         L      - number of layers,
         cache  - storage of intermediate values,
         weights - storage of weights and biases.
    */
    """
    def __init__(self, nx, layers):
        """
        /**
           Initialize the deep neural network.
           nx indicates the number of input features.
           layers is a list that shows the count of nodes in each layer.
           
           Checking conditions:
             - If nx is not an integer, a TypeError is raised.
             - If nx is less than 1, a ValueError is raised.
             - If layers is not a list or is empty, a TypeError is raised.
             - If any element in layers is not a positive integer, a TypeError is raised.
        */
        """
        # Check type of nx
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        # Check that nx exceeds zero
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # Check that layers is a list with at least one element
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        previous = nx
        i = 0  # Only one looping statement is used below
        while i < self.L:
            # Check that each entry is an integer and exceeds zero
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            # He initialization: random values multiplied by sqrt(2 / previous)
            self.weights["W{}".format(i + 1)] = (np.random.randn(layers[i], previous) *
                                                 np.sqrt(2 / previous))
            self.weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            previous = layers[i]
            i += 1
