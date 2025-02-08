#!/usr/bin/env python3
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""
    
    def __init__(self, nx, layers):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if any(not isinstance(n, int) or n <= 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")
        
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        
        input_units = nx
        for l in range(self.L):
            self.weights[f'W{l+1}'] = np.random.randn(layers[l], input_units) * np.sqrt(2 / input_units)
            self.weights[f'b{l+1}'] = np.zeros((layers[l], 1))
            input_units = layers[l]
