#!/usr/bin/env python3
import numpy as np

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        for nodes in layers:
            if type(nodes) is not int or nodes < 1:
                raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        For i in range(self.__L):
            if i == 0:
                self.__weights["W1"] = np.random.randn(layers[0], nx) * np.sqrt(2 / nx)
            else:
                self.__weights["W{}".format(i + 1)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
    
    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

