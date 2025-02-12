#!/usr/bin/env python3
import numpy as np

"""
/** 
   This module defines a deep neural network with private attributes,
   including a method to compute the signal progression using the sigmoid function.
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

    def forward_prop(self, X):
        """
        /**
          Computes the signal progression in the network.
          X is a numpy.ndarray with shape (nx, m) containing input data.
          Saves the input under key A0 in __cache and each activation under key A{l}.
          Uses the sigmoid function at every layer.
          Returns the network output and the cache.
        */
        """
        self.__cache["A0"] = X
        i = 0
        while i < self.__L:
            W = self.__weights["W{}".format(i + 1)]
            b = self.__weights["b{}".format(i + 1)]
            A_prev = self.__cache["A{}".format(i)] if i > 0 else X
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A{}".format(i + 1)] = A
            i += 1
        return A, self.__cache
