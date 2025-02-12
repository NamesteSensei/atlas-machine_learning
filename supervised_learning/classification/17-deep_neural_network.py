#!/usr/bin/env python3
import numpy as np

"""
/** 
   This module defines a deep neural network with all instance attributes set as private.
*/
"""

class DeepNeuralNetwork:
    """
    /**
      Class that defines a deep neural network executing binary classification.
      Private attributes:
        __L      - number of layers,
        __cache  - a storage for intermediate values,
        __weights - a storage for weights and biases.
    */
    """
    def __init__(self, nx, layers):
        """
        /**
          Constructor to initialize the deep neural network.
          nx is the number of input features.
          layers is a list that indicates the number of nodes in each layer.
        */
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev = nx
        i = 0
        while i < self.__L:
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(i + 1)] = (np.random.randn(layers[i], prev) *
                                                   np.sqrt(2 / prev))
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            prev = layers[i]
            i += 1

    @property
    def L(self):
        """
        /**
          Getter to obtain the number of layers.
        */
        """
        return self.__L

    @property
    def cache(self):
        """
        /**
          Getter to obtain the cache storage.
        */
        """
        return self.__cache

    @property
    def weights(self):
        """
        /**
          Getter to obtain the weights storage.
        */
        """
        return self.__weights
