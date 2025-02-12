#!/usr/bin/env python3
import numpy as np

"""
/**
   This module defines a deep neural network updated for multiclass classification.
   Hidden levels use sigmoid activation, while the output level applies softmax.
*/
"""

class DeepNeuralNetwork:
    """
    /**
       Deep neural network class used in multiclass classification.
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
        i = 0
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
           The input is stored with key A0 in __cache.
           Hidden levels use the sigmoid activation.
           The output level applies softmax activation.
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
            if i == self.__L - 1:
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                A = 1 / (1 + np.exp(-Z))
            self.__cache["A{}".format(i + 1)] = A
            i += 1
        return A, self.__cache

    def cost(self, Y, A):
        """
        /**
           Computes cost using categorical cross-entropy.
           Y is a numpy.ndarray with shape (classes, m) of one-hot labels.
           A is a numpy.ndarray with shape (classes, m) of outputs.
           Returns the cost.
        */
        """
        m = Y.shape[1]
        cost_value = -np.sum(Y * np.log(A + 1e-8)) / m
        return cost_value

    def evaluate(self, X, Y):
        """
        /**
           Evaluates the network's predictions.
           X is a numpy.ndarray with shape (nx, m) containing input data.
           Y is a numpy.ndarray with shape (classes, m) of one-hot labels.
           The prediction is the index of the maximum probability.
           Returns the prediction and cost.
        */
        """
        A, _ = self.forward_prop(X)
        cost_value = self.cost(Y, A)
        prediction = np.argmax(A, axis=0).reshape(1, -1)
        return prediction, cost_value

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        /**
           Executes one step of gradient descent to update weights.
           Y is a numpy.ndarray with shape (classes, m) of one-hot labels.
           cache holds intermediate values.
           alpha is the learning rate.
           The derivative at the output level is (A - Y).
           Hidden levels use the sigmoid derivative (A * (1 - A)).
           Updates the private weights.
        */
        """
        m = Y.shape[1]
        i = self.__L
        A_L = cache["A{}".format(i)]
        dZ = A_L - Y
        while i >= 1:
            A_prev = cache["A{}".format(i - 1)]
            W = self.__weights["W{}".format(i)]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights["W{}".format(i)] = W - alpha * dW
            self.__weights["b{}".format(i)] = self.__weights["b{}".format(i)] - alpha * db
            if i > 1:
                A_hidden = cache["A{}".format(i - 1)]
                dZ = (np.matmul(self.__weights["W{}".format(i)].T, dZ) *
                      (A_hidden * (1 - A_hidden)))
            i -= 1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        /**
           Trains the deep neural network.
           X is a numpy.ndarray with shape (nx, m) containing input data.
           Y is a numpy.ndarray with shape (classes, m) of one-hot labels.
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
