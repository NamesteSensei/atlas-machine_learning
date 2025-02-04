#!/usr/bin/env python3
import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Initialize the neuron.

        Parameters:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # Initialize the weights vector using a random normal distribution.
        self.__W = np.random.randn(1, nx)
        # Initialize the bias to 0.
        self.__b = 0
        # Initialize the activated output (prediction) to 0.
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector."""
        return self.__W

    @property
    def b(self):
        """Getter for the bias."""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output."""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Parameters:
            X (numpy.ndarray): Shape (nx, m) containing the input data.

        Returns:
            numpy.ndarray: The activated output of the neuron.
        """
        # Compute the linear transformation: z = W.X + b.
        z = np.matmul(self.__W, X) + self.__b
        # Apply the sigmoid activation function.
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression.

        Parameters:
            Y (numpy.ndarray): Shape (1, m) containing the correct labels.
            A (numpy.ndarray): Shape (1, m) containing the activated outputs.

        Returns:
            float: The cost computed using cross-entropy loss.
        """
        m = Y.shape[1]
        # Compute the cross-entropy cost using a small constant to avoid log(0).
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        Parameters:
            X (numpy.ndarray): Shape (nx, m) containing input data.
            Y (numpy.ndarray): Shape (1, m) containing the correct labels.

        Returns:
            tuple: (predictions, cost) where predictions is a numpy.ndarray of 0's and 1's.
        """
        # Perform forward propagation to obtain the activated output.
        A = self.forward_prop(X)
        # Calculate the cost.
        cost = self.cost(Y, A)
        # Convert activated outputs to binary predictions using threshold 0.5.
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one pass of gradient descent on the neuron.

        Parameters:
            X (numpy.ndarray): Shape (nx, m) containing the input data.
            Y (numpy.ndarray): Shape (1, m) containing the correct labels.
            A (numpy.ndarray): Shape (1, m) containing the activated output.
            alpha (float): The learning rate.

        Updates:
            The private attributes __W and __b are updated.
        """
        m = Y.shape[1]
        # Compute the derivative of the cost with respect to z.
        dz = A - Y
        # Compute the gradient for weights.
        dW = np.matmul(dz, X.T) / m
        # Compute the gradient for the bias.
        db = np.sum(dz) / m
        # Update the weights and bias using the gradients and learning rate.
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron.

        Parameters:
            X (numpy.ndarray): Shape (nx, m) containing the input data.
            Y (numpy.ndarray): Shape (1, m) containing the correct labels.
            iterations (int): The number of iterations to train over.
            alpha (float): The learning rate.

        Raises:
            TypeError: If iterations is not an integer or if alpha is not a float.
            ValueError: If iterations or alpha is not positive.

        Returns:
            tuple: (predictions, cost) after training the neuron.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Perform gradient descent for the specified number of iterations (only one loop allowed).
        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        # Evaluate and return the final predictions and cost.
        return self.evaluate(X, Y)
