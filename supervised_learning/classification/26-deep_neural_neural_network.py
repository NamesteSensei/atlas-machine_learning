#!/usr/bin/env python3
import numpy as np
import pickle

"""
/**
   Deep neural network with persistence methods.
   Allows saving the instance to a pickle file and loading it back.
*/
"""

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers)==0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev = nx
        i = 0  # one loop in __init__
        while i < self.__L:
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(i+1)] = np.random.randn(layers[i], prev)*np.sqrt(2/prev)
            self.__weights["b{}".format(i+1)] = np.zeros((layers[i], 1))
            prev = layers[i]
            i += 1

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
        self.__cache["A0"] = X
        i = 0  # one loop in forward_prop
        while i < self.__L:
            W = self.__weights["W{}".format(i+1)]
            b = self.__weights["b{}".format(i+1)]
            A_prev = self.__cache["A{}".format(i)] if i>0 else X
            Z = np.matmul(W, A_prev)+b
            A = 1/(1+np.exp(-Z))
            self.__cache["A{}".format(i+1)] = A
            i += 1
        return A, self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        return -np.sum(Y*np.log(A)+(1-Y)*np.log(1.0000001-A))/m

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost_val = self.cost(Y, A)
        prediction = np.where(A>=0.5,1,0)
        return prediction, cost_val

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        i = self.__L  # one loop in gradient_descent
        dZ = cache["A{}".format(i)] - Y
        while i>=1:
            A_prev = cache["A{}".format(i-1)]
            W = self.__weights["W{}".format(i)]
            dW = np.matmul(dZ, A_prev.T)/m
            db = np.sum(dZ, axis=1, keepdims=True)/m
            W_copy = W.copy()
            self.__weights["W{}".format(i)] = W - alpha*dW
            self.__weights["b{}".format(i)] = self.__weights["b{}".format(i)] - alpha*db
            if i>1:
                dZ = np.matmul(W_copy.T, dZ) * (A_prev*(1-A_prev))
            i -= 1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations<=0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha<=0:
            raise ValueError("alpha must be positive")
        i = 0  # one loop in train
        while i < iterations:
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            i += 1
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        /**
           Saves the instance to a file in pickle format.
           If filename does not end with ".pkl", the extension is appended.
        */
        """
        if type(filename) is not str:
            return None
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        try:
            with open(filename, "wb") as f:
                pickle.dump(self, f)
        except Exception:
            return None

    @staticmethod
    def load(filename):
        """
        /**
           Loads a pickled DeepNeuralNetwork instance from a file.
           Returns the loaded object or None if unsuccessful.
        */
        """
        try:
            with open(filename, "rb") as f:
                obj = pickle.load(f)
            return obj
        except Exception:
            return None
