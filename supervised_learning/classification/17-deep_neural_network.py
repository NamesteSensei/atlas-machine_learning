import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network for binary classification"""

    def __init__(self, nx, layers):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if any(type(i) is not int or i <= 0 for i in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev_units = nx
        step = 1
        while step <= self.__L:
            self.__weights[f'W{step}'] = (
                np.random.randn(layers[step - 1], prev_units) * np.sqrt(2 / prev_units)
            )
            self.__weights[f'b{step}'] = np.zeros((layers[step - 1], 1))
            prev_units = layers[step - 1]
            step += 1

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        Perform forward propagation using sigmoid activation function.
        X: numpy.ndarray - input data of shape (nx, m)
        Returns: the activated output and the cache
        """
        self.__cache["A0"] = X  # Store input data
        step = 1
        while step <= self.__L:
            W = self.__weights[f'W{step}']
            b = self.__weights[f'b{step}']
            A_prev = self.__cache[f'A{step - 1}']

            Z = np.dot(W, A_prev) + b  # Linear step
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache[f'A{step}'] = A  # Store activation in cache

            step += 1

        return A, self.__cache
