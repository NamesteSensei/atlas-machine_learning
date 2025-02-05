#!/usr/bin/env python3

"""
Module: 14-neural_network
This module implements a neural network with one hidden layer
for binary classification, including forward propagation, cost calculation,
evaluation of predictions, gradient descent optimization, and training.

Classes:
    NeuralNetwork: Defines a neural network with one hidden layer.
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """Neural network with one hidden layer for binary classification."""

    def __init__(self, nx, nodes):
        """Initialize the neural network.

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.

        Raises:
            TypeError: If nx or nodes is not an integer.
            ValueError: If nx or nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize private attributes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = np.zeros((1, 1))
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """Performs forward propagation using the sigmoid activation function.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            tuple: Activated outputs (A1, A2) of the hidden and output layers.
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost using logistic regression loss function.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Activated output of the neuron.

        Returns:
            float: Computed cost.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).

        Returns:
            tuple: (predictions, cost) where predictions are classified labels.
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        predictions = np.where(A2 >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Performs one pass of gradient descent to update weights and biases.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A1 (numpy.ndarray): Activated output of the hidden layer.
            A2 (numpy.ndarray): Activated output of the output neuron.
            alpha (float): Learning rate.
        """
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network using gradient descent.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            iterations (int): Number of training iterations.
            alpha (float): Learning rate.

        Returns:
            tuple: (predictions, cost) after training.
        """
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
