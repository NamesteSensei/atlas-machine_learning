#!/usr/bin/env python3

import numpy as np
from deep_neural_network import DeepNeuralNetwork

# Testing invalid cases
try:
    DeepNeuralNetwork("invalid", [5, 3, 1])
except Exception as e:
    print(e)  # Expected: nx must be an integer

try:
    DeepNeuralNetwork(0, [5, 3, 1])
except Exception as e:
    print(e)  # Expected: nx must be a positive integer

try:
    DeepNeuralNetwork(1.5, [5, 3, 1])
except Exception as e:
    print(e)  # Expected: nx must be an integer

try:
    DeepNeuralNetwork(5, "invalid")
except Exception as e:
    print(e)  # Expected: layers must be a list of positive integers

try:
    DeepNeuralNetwork(5, [])
except Exception as e:
    print(e)  # Expected: layers must be a list of positive integers

try:
    DeepNeuralNetwork(5, [5, -3, 1])
except Exception as e:
    print(e)  # Expected: layers must be a list of positive integers

try:
    DeepNeuralNetwork(5, [5, 3, "invalid"])
except Exception as e:
    print(e)  # Expected: layers must be a list of positive integers
