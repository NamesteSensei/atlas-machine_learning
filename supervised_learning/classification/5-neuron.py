#!/usr/bin/env python3
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Initialize Neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """Performs forward propagation"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        A = self.forward_prop(X)
        predictions = (A >= 0.5).astype(int)  # Vectorized decision
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Performs one step of gradient descent to update weights"""
        m = X.shape[1]
        dZ = A - Y  # Error term
        dW = np.matmul(dZ, X.T) / m  # Compute gradients (vectorized)
        db = np.mean(dZ)  # Compute bias gradient

        self.__W -= alpha * dW  # Update weights
        self.__b -= alpha * db  # Update bias
