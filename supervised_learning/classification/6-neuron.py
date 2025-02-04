#!/usr/bin/env python3
"""
Neuron class for binary classification with training functionality.
"""

import numpy as np


class Neuron:
    """
    Represents a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Initialize the neuron.

        Parameters:
        nx (int): Number of input features.

        Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)  # Initialize weights
        self.__b = 0  # Initialize bias
        self.__A = 0  # Initialize activation output

    @property
    def W(self):
        """Returns weights."""
        return self.__W

    @property
    def b(self):
        """Returns bias."""
        return self.__b

    @property
    def A(self):
        """Returns activated output."""
        return self.__A

    def forward_prop(self, X):
        """
        Perform forward propagation.

        Parameters:
        X (numpy.ndarray): Shape (nx, m), Input data.

        Returns:
        numpy.ndarray: Activated output.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
        return self.__A

    def cost(self, Y, A):
        """
        Compute logistic regression cost.

        Parameters:
        Y (numpy.ndarray): Correct labels.
        A (numpy.ndarray): Activated output.

        Returns:
        float: Cost value.
        """
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m

    def evaluate(self, X, Y):
        """
        Evaluate the neuron's predictions.

        Parameters:
        X (numpy.ndarray): Input data.
        Y (numpy.ndarray): Correct labels.

        Returns:
        tuple: (Predictions, cost).
        """
        A = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)  # Threshold at 0.5
        return predictions, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Perform one step of gradient descent.

        Parameters:
        X (numpy.ndarray): Input data.
        Y (numpy.ndarray): Correct labels.
        A (numpy.ndarray): Activated output.
        alpha (float): Learning rate.

        Returns:
        None
        """
        m = X.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.matmul(dZ, X.T)
        db = (1 / m) * np.sum(dZ)

        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Train the neuron using gradient descent.

        Parameters:
        X (numpy.ndarray): Input data.
        Y (numpy.ndarray): Correct labels.
        iterations (int): Number of iterations.
        alpha (float): Learning rate.

        Returns:
        tuple: (Final predictions, cost history list).
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, (float, int)):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        cost_history = []

        # **Single Loop to Meet Requirements**
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

            if i % 100 == 0 or i == iterations - 1:
                cost = self.cost(Y, A)
                cost_history.append(float(cost))  # Store as float
                print(f"Iteration {i}/{iterations} - Cost: {cost:.6f}")

        return self.evaluate(X, Y)[0], cost_history
