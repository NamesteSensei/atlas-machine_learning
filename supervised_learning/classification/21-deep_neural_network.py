#!/usr/bin/env python3
import numpy as np

class DeepNeuralNetwork:
    """ Defines a deep neural network performing binary classification. """

    def __init__(self, nx, layers):
        """ Initialize the network """
        np.random.seed(0)  # Set fixed seed for consistency

        if type(nx) is not int or nx < 1:
            raise ValueError("nx must be a positive integer")

        if (type(layers) is not list or len(layers) == 0 or 
            not all(isinstance(i, int) and i > 0 for i in layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)  # Number of layers
        self.__cache = {}
        self.__weights = {}

        prev = nx
        for i in range(self.__L):  # 🔴 Loop 1: Weight Initialization
            self.__weights["W{}".format(i + 1)] = np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            prev = layers[i]

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
        """ Computes forward propagation """
        self.__cache["A0"] = X
        for i in range(self.__L):  # 🔴 Loop 2: Forward Propagation
            W = self.__weights["W{}".format(i + 1)]
            b = self.__weights["b{}".format(i + 1)]
            A_prev = self.__cache["A{}".format(i)]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache["A{}".format(i + 1)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """ Computes cost using logistic regression """
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m

    def evaluate(self, X, Y):
        """ Evaluates the network's predictions """
        A, _ = self.forward_prop(X)
        cost_val = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost_val

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Performs gradient descent weight updates """
        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y
        for i in range(self.__L, 0, -1):  # 🔴 Loop 3: Gradient Descent Updates
            A_prev = cache["A{}".format(i - 1)]
            dW = np.matmul(dZ, A_prev.T) / m
