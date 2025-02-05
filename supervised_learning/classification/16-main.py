#!/usr/bin/env python3
"""
Test script for the DeepNeuralNetwork class (task 16).
This script tests various input scenarios:
    - Normal usage.
    - Edge cases for input validation.
"""

import numpy as np
DeepNN = __import__('16-deep_neural_network').DeepNeuralNetwork

# ✅ Test 1: Normal usage
print("✅ Test 1: Normal usage")
try:
    dnn = DeepNN(5, [3, 2, 1])
    print("L:", dnn.L)
    print("weights keys:", list(dnn.weights.keys()))
except Exception as e:
    print("Error:", e)

# ✅ Test 2: nx is a valid integer (should work)
print("\n✅ Test 2: nx is a valid integer")
try:
    dnn = DeepNN(1, [2])
    print("L:", dnn.L)
except Exception as e:
    print("Error:", e)

# ✅ Test 3: nx is a float (error expected)
print("\n❌ Test 3: nx is a float")
try:
    dnn = DeepNN(3.5, [2])
except Exception as e:
    print("Error:", e)

# ✅ Test 4: nx is zero (error expected)
print("\n❌ Test 4: nx is 0")
try:
    dnn = DeepNN(0, [2])
except Exception as e:
    print("Error:", e)

# ✅ Test 5: Layers with a single element (should work)
print("\n✅ Test 5: Layers with a single element")
try:
    dnn = DeepNN(5, [4])
    print("L:", dnn.L)
except Exception as e:
    print("Error:", e)

# ✅ Test 6: Layers is not a list (error expected)
print("\n❌ Test 6: Layers is not a list")
try:
    dnn = DeepNN(5, "not a list")
except Exception as e:
    print("Error:", e)

# ✅ Test 7: Layers is an empty list (error expected)
print("\n❌ Test 7: Layers is an empty list")
try:
    dnn = DeepNN(5, [])
except Exception as e:
    print("Error:", e)

# ✅ Test 8: Layers contains a non-positive integer (error expected)
print("\n❌ Test 8: Layers contains a non-positive integer")
try:
    dnn = DeepNN(5, [3, -1, 2])
except Exception as e:
    print("Error:", e)

