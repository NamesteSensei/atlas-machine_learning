#!/usr/bin/env python3
"""
Module that defines the NeuralNetwork class for binary classification.
This network is like a team of salespeople (hidden layer) that feed a manager
(output neuron) who decides if a customer buys (1) or not (0).
"""

import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer for binary classification.

    Private instance attributes:
        __W1: Weights for the hidden layer (random normal init).
        __b1: Biases for the hidden layer (zeros).
        __A1: Activated output for the hidden layer.
        __W2: Weights for the output neuron (random normal init).
        __b2: Bias for the output neuron (0).
        __A2: Activated output for the output neuron.
    """

    def __init__(self, nx, nodes):
        """
        Initialize the neural network.

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If nodes is not an integer.
            ValueError: If nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize hidden layer weights with random normal values.
        self.__W1 = np.random.randn(nodes, nx)
        # Initialize hidden layer biases as zeros.
        self.__b1 = np.zeros((nodes, 1))
        # Activated output for hidden layer starts at 0.
        self.__A1 = 0

        # Initialize output neuron weights with random normal values.
        self.__W2 = np.random.randn(1, nodes)
        # Initialize output neuron bias to 0.
        self.__b2 = 0
        # Activated output for output neuron starts at 0.
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for hidden layer weights."""
        return self.__W1

    @property
    def b1(self):
        """Getter for hidden layer biases."""
        return self.__b1

    @property
    def A1(self):
        """Getter for hidden layer activated output."""
        return self.__A1

    @property
    def W2(self):
        """Getter for output neuron weights."""
        return self.__W2

    @property
    def b2(self):
        """Getter for output neuron bias."""
        return self.__b2

    @property
    def A2(self):
        """Getter for output neuron activated output."""
        return self.__A2

    def forward_prop(self, X):
        """
        Compute the forward propagation of the network.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            tuple: (A1, A2) activated outputs.
        """
        # Compute Z1 = W1 * X + b1 for the hidden layer.
        Z1 = np.matmul(self.__W1, X) + self.__b1
        # Apply sigmoid activation for the hidden layer.
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Compute Z2 = W2 * A1 + b2 for the output neuron.
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        # Apply sigmoid activation for the output neuron.
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Compute the cost using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels of shape (1, m).
            A (numpy.ndarray): Activated output of shape (1, m).

        Returns:
            float: Logistic regression cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the network's predictions.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).

        Returns:
            tuple: (prediction, cost) where prediction is a binary array.
        """
        self.forward_prop(X)
        # Convert probabilities to binary predictions.
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        cost_val = self.cost(Y, self.__A2)
        return prediction, cost_val

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Perform one pass of gradient descent.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).
            A1 (numpy.ndarray): Activated hidden layer output.
            A2 (numpy.ndarray): Activated output neuron output.
            alpha (float): Learning rate.
        """
        m = Y.shape[1]
        # Derivative for output layer.
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        # Derivative for hidden layer.
        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Update parameters.
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Train the network via gradient descent.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).
            iterations (int): Number of iterations to train.
            alpha (float): Learning rate.

        Raises:
            TypeError: If iterations is not an integer.
            ValueError: If iterations is not positive.
            TypeError: If alpha is not a float.
            ValueError: If alpha is not positive.

        Returns:
            tuple: (prediction, cost) after training.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
        return self.evaluate(X, Y)
