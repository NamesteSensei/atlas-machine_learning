#!/usr/bin/env python3

"""
Module: 6-neuron
This module implements a single neuron performing binary classification.
It includes forward propagation, cost calculation, evaluation,
and gradient descent for training the neuron.

Classes:
    Neuron: Defines a single neuron performing binary classification.
"""

import numpy as np


class Neuron:
    """Neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize the neuron.

        Args:
            nx (int): Number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
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
        """Get the weights of the neuron."""
        return self.__W

    @property
    def b(self):
        """Get the bias of the neuron."""
        return self.__b

    @property
    def A(self):
        """Get the activated output of the neuron."""
        return self.__A

    def forward_prop(self, X):
        """Perform forward propagation using sigmoid activation.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Activated output.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculate cost using logistic regression loss function.

        Args:
            Y (numpy.ndarray): Correct labels.
            A (numpy.ndarray): Activated output.

        Returns:
            float: Cost of the prediction.
        """
        m = Y.shape[1]
        cost = (-np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate predictions and compute cost.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Correct labels.

        Returns:
            tuple: Predicted labels and cost.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Perform gradient descent to update weights and bias.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Correct labels.
            A (numpy.ndarray): Activated output.
            alpha (float): Learning rate.
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Train neuron by optimizing weights using gradient descent.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Correct labels.
            iterations (int): Number of training iterations.
            alpha (float): Learning rate.

        Returns:
            tuple: Predicted labels and cost after training.

        Raises:
            TypeError: If iterations is not an integer or alpha is not a float.
            ValueError: If iterations is less than 1 or alpha is not positive.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        return self.evaluate(X, Y)
