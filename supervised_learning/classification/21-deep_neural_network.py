#!/usr/bin/env python3
import numpy as np

"""
/** 
   This module defines a deep neural network with private attributes,
   including methods to compute signal progression, cost, evaluation, and weight updates via gradient descent.
*/
"""

class DeepNeuralNetwork:
    """
    /**
      Class that defines a deep neural network executing binary classification.
      Private attributes:
        __L      - number of layers,
        __cache  - a storage for intermediate values,
        __weights - a storage for weights and biases.
    */
    """
    def __init__(self, nx, layers):
        """
          Constructor to initialize the deep neural network.
          nx is the number of input features.
          layers is a list that indicates the number of nodes in each layer.
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
        prev = nx
        for i in range(self.__L):  # 🔴 Loop 1: Weight Initialization
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(i + 1)] = np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            prev = layers[i]

    @property
    def L(self):
        """
          Getter to obtain the number of layers.
        """
        return self.__L

    @property
    def cache(self):
        """
          Getter to obtain the cache storage.
        """
        return self.__cache

    @property
    def weights(self):
        """
          Getter to obtain the weights storage.
        """
        return self.__weights

    def forward_prop(self, X):
        """
          Computes the signal progression in the network.
          X is a numpy.ndarray with shape (nx, m) containing input data.
          Saves the input under key A0 in __cache and each activation under key A{l}.
          Uses the sigmoid function at every layer.
          Returns the network output and the cache.
        """
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
        """
          Computes cost using logistic regression.
          Y is a numpy.ndarray with shape (1, m) containing true labels.
          A is a numpy.ndarray with shape (1, m) containing the network output.
          Returns the cost.
        """
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m

    def evaluate(self, X, Y):
        """
          Computes the network prediction and cost.
          X is a numpy.ndarray with shape (nx, m) containing input data.
          Y is a numpy.ndarray with shape (1, m) containing true labels.
          A prediction is 1 when output is at least 0.5; else 0.
          Returns the prediction and cost.
        """
        A, _ = self.forward_prop(X)
        cost_val = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost_val

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
          Executes one cycle of gradient descent to adjust weights.
          Y is a numpy.ndarray with shape (1, m) containing true labels.
          cache is a storage containing intermediate values.
          alpha is the learning rate.
          The output-layer derivative is computed as (A – Y).
          Hidden layers use the sigmoid derivative (A * (1 – A)).
          Updates the private weights.
        """
        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y
        for i in range(self.__L, 0, -1):  # 🔴 Loop 3: Gradient Descent
            A_prev = cache["A{}".format(i - 1)]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights["W{}".format(i)] -= alpha * dW
            self.__weights["b{}".format(i)] -= alpha * db
            if i > 1:
                A = cache["A{}".format(i - 1)]
                dZ = np.matmul(self.__weights["W{}".format(i)].T, dZ) * (A * (1 - A))

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
          Trains the deep neural network.
          X is a numpy.ndarray with shape (nx, m) containing input data.
          Y is a numpy.ndarray with shape (1, m) containing true labels.
          iterations is the number of training cycles.
          alpha is the learning rate.
          Raises exceptions if iterations or alpha are invalid.
          Returns the evaluation on the training data.
        """
        if type(iterations) is not int or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float or alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):  # Calls forward_prop and gradient_descent within loop
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
        
        return self.evaluate(X, Y)
