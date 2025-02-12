#!/usr/bin/env python3
import numpy as np

"""
/**
   This module defines a deep neural network with private attributes,
   including a method that trains the network.
   The network uses gradient descent to update weights.
*/
"""

class DeepNeuralNetwork:
    """
    /**
       Deep neural network class used in binary classification.
       Private attributes:
         __L       - count of levels,
         __cache   - storage of intermediate values,
         __weights - storage of weights and biases.
    */
    """
    def __init__(self, nx, layers):
        """
        /**
           Initialize the deep neural network.
           nx indicates the count of input features.
           layers is a list showing the count of nodes in each level.
           Raises exceptions if nx is not an integer, if nx is less than one,
           if layers is not a list or is empty, or if any element in layers is not a positive integer.
        */
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        previous = nx
        i = 0  # Only one looping statement will be used
        while i < self.__L:
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(i + 1)] = (np.random.randn(layers[i], previous) *
                                                   np.sqrt(2 / previous))
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            previous = layers[i]
            i += 1

    @property
    def L(self):
        """
        /**
           Return the count of levels.
        */
        """
        return self.__L

    @property
    def cache(self):
        """
        /**
           Return the storage of intermediate values.
        */
        """
        return self.__cache

    @property
    def weights(self):
        """
        /**
           Return the storage of weights.
        */
        """
        return self.__weights

    def forward_prop(self, X):
        """
        /**
           Computes the progression of signals in the network.
           X is a numpy.ndarray with shape (nx, m) containing input data.
           The input is stored in __cache with key A0; each subsequent activation is stored with key A{level}.
           The sigmoid activation is applied at every level.
           Returns the final output and the cache.
        */
        """
        self.__cache["A0"] = X
        i = 0
        while i < self.__L:
            W = self.__weights["W{}".format(i + 1)]
            b = self.__weights["b{}".format(i + 1)]
            A_prev = self.__cache["A{}".format(i)] if i > 0 else X
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A{}".format(i + 1)] = A
            i += 1
        return A, self.__cache

    def cost(self, Y, A):
        """
        /**
           Computes cost using logistic regression.
           Y is a numpy.ndarray with shape (1, m) of true labels.
           A is a numpy.ndarray with shape (1, m) of network outputs.
           Returns the cost.
        */
        """
        m = Y.shape[1]
        cost_value = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost_value

    def evaluate(self, X, Y):
        """
        /**
           Evaluates the network's predictions.
           X is a numpy.ndarray with shape (nx, m) of input data.
           Y is a numpy.ndarray with shape (1, m) of true labels.
           A prediction is 1 if the output is at least 0.5; else 0.
           Returns the prediction and cost.
        */
        """
        A, _ = self.forward_prop(X)
        cost_value = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost_value

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        /**
           Executes one step of gradient descent to update weights.
           Y is a numpy.ndarray with shape (1, m) of true labels.
           cache holds the intermediate values.
           alpha is the learning rate.
           The derivative at the output level is (A - Y).
           Hidden levels use the sigmoid derivative (A * (1 - A)).
           Updates the private weights.
        */
        """
        m = Y.shape[1]
        i = self.__L
        dZ = cache["A{}".format(i)] - Y
        while i >= 1:
            A_prev = cache["A{}".format(i - 1)]
            W = self.__weights["W{}".format(i)]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights["W{}".format(i)] = W - alpha * dW
            self.__weights["b{}".format(i)] = self.__weights["b{}".format(i)] - alpha * db
            if i > 1:
                A_current = cache["A{}".format(i - 1)]
                dZ = (np.matmul(self.__weights["W{}".format(i)].T, dZ) *
                      (A_current * (1 - A_current)))
            i -= 1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        /**
           Trains the deep neural network.
           X is a numpy.ndarray with shape (nx, m) containing input data.
           Y is a numpy.ndarray with shape (1, m) containing true labels.
           iterations indicates the number of training cycles.
           alpha is the learning rate.
           Raises exceptions if iterations is not an integer or if alpha is not a float or positive.
           Returns the evaluation on the training data.
        */
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        i = 0
        while i < iterations:
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            i += 1
        return self.evaluate(X, Y)
