#!/usr/bin/env python3
"""
Neuron class performing binary classification with multiple epochs.
"""

import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """
    Class that represents a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Initialize the neuron.

        Parameters:
        nx -- int: Number of input features.
        
        Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)  # Random weight initialization
        self.__b = 0  # Bias initialized to 0
        self.__A = 0  # Activated output of the neuron

    @property
    def W(self):
        """Returns the weights."""
        return self.__W

    @property
    def b(self):
        """Returns the bias."""
        return self.__b

    @property
    def A(self):
        """Returns the activated output."""
        return self.__A

    def forward_prop(self, X):
        """
        Perform forward propagation.

        Parameters:
        X -- numpy.ndarray: Shape (nx, m), Input data

        Returns:
        numpy.ndarray: Shape (1, m), Activated output
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
        return self.__A

    def cost(self, Y, A):
        """
        Compute the cost using logistic regression loss.

        Parameters:
        Y -- numpy.ndarray: Shape (1, m), Correct labels
        A -- numpy.ndarray: Shape (1, m), Activated output
        
        Returns:
        float: Cost value
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neuron's predictions.

        Parameters:
        X -- numpy.ndarray: Shape (nx, m), Input data
        Y -- numpy.ndarray: Shape (1, m), Correct labels
        
        Returns:
        tuple: (Predictions, cost)
        """
        A = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)  # Thresholding at 0.5
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Perform one step of gradient descent.

        Parameters:
        X -- numpy.ndarray: Shape (nx, m), Input data
        Y -- numpy.ndarray: Shape (1, m), Correct labels
        A -- numpy.ndarray: Shape (1, m), Activated output
        alpha -- float: Learning rate
        """
        m = X.shape[1]  # Number of examples
        dZ = A - Y  # Error term
        dW = (1 / m) * np.matmul(dZ, X.T)  # Gradient for weights
        db = (1 / m) * np.sum(dZ)  # Gradient for bias
        
        # Update weights and bias
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Train the neuron over multiple epochs using gradient descent.

        Parameters:
        X -- numpy.ndarray: Shape (nx, m), Input data
        Y -- numpy.ndarray: Shape (1, m), Correct labels
        iterations -- int: Number of iterations
        alpha -- float: Learning rate

        Returns:
        tuple: (Final predictions, cost history list)
        """
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("alpha must be a positive float")

        cost_history = []  # Ensure this is a list

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            
            cost = self.cost(Y, A)  # Always compute cost
            if i % 100 == 0 or i == iterations - 1:
                cost_history.append(float(cost))  # Append cost as a float
                print(f"Iteration {i}/{iterations} - Cost: {cost:.6f}")

        return self.evaluate(X, Y)[0], cost_history  # Return list of costs
