#!/usr/bin/env python3
"""
Test script for the DeepNeuralNetwork class (task 16).
This script tests various input scenarios:
    - Normal usage.
    - nx is 1.
    - nx is a float.
    - nx is 0.
    - layers is a list of 1 element.
    - layers is not a list.
    - layers is an empty list.
    - layers contains an element that is not a positive integer.
Each test prints the result or the error message.
"""

import numpy as np
DeepNN = __import__('16-deep_neural_network').DeepNeuralNetwork

# Test 1: Normal usage
print("Test 1: Normal usage")
try:
    dnn = DeepNN(5, [3, 2, 1])
    print("L:", dnn.L)
    print("weights keys:", list(dnn.weights.keys()))
except Exception as e:
    print("Error:", e)

# Test 2: nx is 1 (should work)
print("\nTest 2: nx is 1")
try:
    dnn = DeepNN(1, [2])
    print("L:", dnn.L)
except Exception as e:
    print("Error:", e)

# Test 3: nx is a float (error)
print("\nTest 3: nx is a float")
try:
    dnn = DeepNN(3.5, [2])
except Exception as e:
    print("Error:", e)

# Test 4: nx is 0 (error)
print("\nTest 4: nx is 0")
try:
    dnn = DeepNN(0, [2])
except Exception as e:
    print("Error:", e)

# Test 5: layers is a list of 1 element (should work)
print("\nTest 5: layers is a list of 1 element")
try:
    dnn = DeepNN(5, [4])
    print("L:", dnn.L)
except Exception as e:
    print("Error:", e)

# Test 6: layers is not a list (error)
print("\nTest 6: layers is not a list")
try:
    dnn = DeepNN(5, "not a list")
except Exception as e:
    print("Error:", e)

# Test 7: layers is an empty list (error)
print("\nTest 7: layers is an empty list")
try:
    dnn = DeepNN(5, [])
except Exception as e:
    print("Error:", e)

# Test 8: layers contains an element that is not a positive integer (error)
print("\nTest 8: layers contains an element that is not a positive integer")
try:
    dnn = DeepNN(5, [3, -1, 2])
except Exception as e:
    print("Error:", e)
