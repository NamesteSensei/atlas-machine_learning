#!/usr/bin/env python3
import numpy as np

"""
/**
   Deep neural network for multiclass classification with selectable hidden-layer activation.
   Hidden layers use the activation specified by the parameter ('sig' for sigmoid or 'tanh' for hyperbolic tangent);
   the output layer always uses softmax.
*/
"""

class DeepNeuralNetwork:
    def __init__(self, nx, layers, activation='sig'):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers)==0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
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
    @property
    def activation(self):
        return self.__activation

    def forward_prop(self, X):
        self.__cache["A0"] = X
        i = 0  # one loop in forward_prop
        while i < self.__L:
            W = self.__weights["W{}".format(i+1)]
            b = self.__weights["b{}".format(i+1)]
            A_prev = self.__cache["A{}".format(i)] if i>0 else X
            Z = np.matmul(W, A_prev) + b
            if i == self.__L - 1:
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                if self.__activation == 'sig':
                    A = 1/(1+np.exp(-Z))
                else:
                    A = np.tanh(Z)
            self.__cache["A{}".format(i+1)] = A
            i += 1
        return A, self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        return -np.sum(Y*np.log(A+1e-8))/m

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost_val = self.cost(Y, A)
        prediction = np.argmax(A, axis=0).reshape(1,-1)
        return prediction, cost_val

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        i = self.__L  # one loop in gradient_descent
        dZ = cache["A{}".format(i)] - Y
        while i >= 1:
            A_prev = cache["A{}".format(i-1)]
            W = self.__weights["W{}".format(i)]
            dW = np.matmul(dZ, A_prev.T)/m
            db = np.sum(dZ, axis=1, keepdims=True)/m
            W_copy = W.copy()
            self.__weights["W{}".format(i)] = W - alpha*dW
            self.__weights["b{}".format(i)] = self.__weights["b{}".format(i)] - alpha*db
            if i > 1:
                if self.__activation == 'sig':
                    dZ = np.matmul(W_copy.T, dZ) * (A_prev*(1-A_prev))
                else:
                    dZ = np.matmul(W_copy.T, dZ) * (1 - np.power(A_prev, 2))
            i -= 1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations<=0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha<=0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step<=0 or step>iterations:
                raise ValueError("step must be positive and <= iterations")
        cost_list = []
        step_list = []
        i = 0  # one loop in train
        while i <= iterations:
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost_val = self.cost(Y, A)
                cost_list.append(cost_val)
                step_list.append(i)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost_val))
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)
            i += 1
        # Note: graph plotting is handled in the test file.
        self.__cache["cost_list"] = cost_list
        self.__cache["step_list"] = step_list
        return self.evaluate(X, Y)
