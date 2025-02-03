#!/usr/bin/env python3
"""Neuron performing binary classification"""

import numpy as np


class Neuron:
    """Defines a single neuron for binary classification"""

    def __init__(self, nx):
        """
        Initializes a neuron

        Args:
            nx (int): Number of input features

        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Returns the weights of the neuron"""
        return self.__W

    @property
    def b(self):
        """Returns the bias of the neuron"""
        return self.__b

    @property
    def A(self):
        """Returns the activated output of the neuron"""
        return self.__A

    def forward_prop(self, X):
        """
        Performs forward propagation

        Args:
            X (numpy.ndarray): Input data of shape (nx, m)

        Returns:
            numpy.ndarray: Activated output of the neuron
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Computes the cost using logistic regression

        Args:
            Y (numpy.ndarray): True labels of shape (1, m)
            A (numpy.ndarray): Activated output of shape (1, m)

        Returns:
            float: Cost function result
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) *
                       np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions

        Args:
            X (numpy.ndarray): Input data of shape (nx, m)
            Y (numpy.ndarray): True labels of shape (1, m)

        Returns:
            tuple: Predicted labels and cost of the network
        """
        A = self.forward_prop(X)
        predictions = (A >= 0.5).astype(int)  # Vectorized decision
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs gradient descent to update weights and bias

        Args:
            X (numpy.ndarray): Input data of shape (nx, m)
            Y (numpy.ndarray): True labels of shape (1, m)
            A (numpy.ndarray): Activated output of shape (1, m)
            alpha (float): Learning rate

        Returns:
            None
        """
        m = X.shape[1]
        dZ = A - Y  # Error term
        dW = np.matmul(dZ, X.T) / m  # Compute gradients
        db = np.mean(dZ)  # Compute bias gradient

        self.__W -= alpha * dW  # Update weights
        self.__b -= alpha * db  # Update bias
