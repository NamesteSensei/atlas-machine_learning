#!/usr/bin/env python3
import numpy as np

class DeepNeuralNetwork:
    """
    DeepNeuralNetwork class defines a deep neural network performing binary classification.
    It includes initialization, forward propagation, cost calculation, evaluation, and gradient descent.
    """
    
    def __init__(self, nx, layers):
        """
        Initialize the deep neural network.

        Parameters:
        /** nx: number of input features.
         *  layers: list of nodes in each layer.
         *
         * Validations:
         *  - nx must be an integer and >= 1.
         *  - layers must be a non-empty list of positive integers.
         */
        if type(nx) is not int or nx < 1:
            raise TypeError("nx must be an integer and >= 1")
        if type(layers) is not list or len(layers) == 0 or \
           not all(isinstance(i, int) and i > 0 for i in layers):
            raise TypeError("layers must be a list of positive integers")
        
        self.__L = len(layers)      /** Total number of layers **/
        self.__cache = {}           /** Cache to hold all intermediary activations **/
        self.__weights = {}         /** Dictionary to hold weights and biases **/
        
        prev = nx                /** Number of nodes in the previous layer (input size) **/
        i = 1
        while i <= self.__L:
            /** He initialization for weights for layer i **/
            self.__weights["W" + str(i)] = np.random.randn(layers[i - 1], prev) * np.sqrt(2 / prev)
            /** Biases initialized to zeros for layer i **/
            self.__weights["b" + str(i)] = np.zeros((layers[i - 1], 1))
            prev = layers[i - 1]   /** Update previous layer's node count **/
            i += 1
    
    @property
    def L(self):
        """Return the number of layers."""
        return self.__L
    
    @property
    def cache(self):
        """Return the cache dictionary."""
        return self.__cache
    
    @property
    def weights(self):
        """Return the weights dictionary."""
        return self.__weights
    
    def forward_prop(self, X):
        """
        Conduct forward propagation through the network.

        Parameters:
        /** X: numpy.ndarray of shape (nx, m) containing the input data.
         *
         * Returns:
         *  The output of the network and the cache dictionary.
         */
        self.__cache["A0"] = X
        i = 1
        while i <= self.__L:
            W = self.__weights["W" + str(i)]         /** Get current layer's weights **/
            b = self.__weights["b" + str(i)]         /** Get current layer's biases **/
            A_prev = self.__cache["A" + str(i - 1)]    /** Activation from previous layer **/
            Z = np.matmul(W, A_prev) + b             /** Compute linear combination **/
            self.__cache["A" + str(i)] = 1 / (1 + np.exp(-Z))  /** Apply sigmoid activation **/
            i += 1
        return self.__cache["A" + str(self.__L)], self.__cache
    
    def cost(self, Y, A):
        """
        Calculate the cost using logistic regression.

        Parameters:
        /** Y: numpy.ndarray of shape (1, m) containing true labels.
         *  A: numpy.ndarray of shape (1, m) containing the activated output.
         *
         * Returns:
         *  The logistic regression cost.
         */
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost
    
    def evaluate(self, X, Y):
        """
        Evaluate the network's predictions.

        Parameters:
        /** X: numpy.ndarray of shape (nx, m) containing the input data.
         *  Y: numpy.ndarray of shape (1, m) containing true labels.
         *
         * Returns:
         *  A tuple (predictions, cost) where predictions is a numpy.ndarray of predicted labels.
         */
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost
    
    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Perform one pass of gradient descent on the network.

        Parameters:
        /** Y: numpy.ndarray of shape (1, m) containing the true labels.
         *  cache: dictionary containing intermediary values from forward propagation.
         *  alpha: learning rate.
         *
         * Updates the private attribute __weights using the gradients computed.
         */
        m = Y.shape[1]
        dZ = cache["A" + str(self.__L)] - Y   /** Compute error at the output layer **/
        i = self.__L
        while i >= 1:
            A_prev = cache["A" + str(i - 1)]
            dW = np.matmul(dZ, A_prev.T) / m    /** Gradient of weights **/
            db = np.sum(dZ, axis=1, keepdims=True) / m  /** Gradient of biases **/
            /** Update current layer's weights and biases **/
            self.__weights["W" + str(i)] -= alpha * dW
            self.__weights["b" + str(i)] -= alpha * db
            if i > 1:
                /** Compute the gradient for the previous layer using backpropagation **/
                dZ = np.matmul(self.__weights["W" + str(i)].T, dZ) * (A_prev * (1 - A_prev))
            i -= 1
