#!/usr/bin/env python3
import numpy as np

"""
Deep Neural Network performing binary classification.
"""

class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Constructor to initialize the deep neural network."""
        if type(nx) is not int or nx < 1:
            raise TypeError("nx must be an integer and >= 1")
        if type(layers) is not list or len(layers) == 0 or not all(isinstance(i, int) and i > 0 for i in layers):
            raise TypeError("layers must be a list of positive integers")
        
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        
        prev = nx
        for i in range(1, self.__L + 1):  # Loop for layer initialization
            self.__weights[f"W{i}"] = np.random.randn(layers[i - 1], prev) * np.sqrt(2 / prev)
            self.__weights[f"b{i}"] = np.zeros((layers[i - 1], 1))
            prev = layers[i - 1]

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Computes the forward propagation and stores cache values."""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):  # Loop for forward propagation
            W = self.__weights[f"W{i}"]
            b = self.__weights[f"b{i}"]
            A_prev = self.__cache[f"A{i-1}"]
            Z = np.matmul(W, A_prev) + b
            self.__cache[f"A{i}"] = 1 / (1 + np.exp(-Z))  # Sigmoid activation
        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """Computes the logistic regression cost."""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions."""
        A, _ = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Performs one pass of gradient descent to update weights."""
        m = Y.shape[1]
        dZ = cache[f"A{self.__L}"] - Y  # Error at output layer

        for i in range(self.__L, 0, -1):  # Loop for backpropagation
            A_prev = cache[f"A{i-1}"]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # Update weights
            self.__weights[f"W{i}"] -= alpha * dW
            self.__weights[f"b{i}"] -= alpha * db

            if i > 1:  # Skip input layer
                A = cache[f"A{i-1}"]
                dZ = np.matmul(self.__weights[f"W{i}"].T, dZ) * (A * (1 - A))  # Sigmoid derivative
