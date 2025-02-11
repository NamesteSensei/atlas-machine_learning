#!/usr/bin/env python3
import numpy as np

# Import the deep network class from our module
Deep = __import__('16-deep_neural_network').DeepNeuralNetwork

def run_tests():
    print("Normal condition test")
    try:
        dn = Deep(5, [3, 1, 2])
        print("cache:", dn.cache)
        print("weights:", dn.weights)
        print("L:", dn.L)
    except Exception as err:
        print("Error:", err)
    
    print("nx equals one test")
    try:
        dn = Deep(1, [3, 2])
        print("L:", dn.L)
    except Exception as err:
        print("Error:", err)
    
    print("nx is a non–integer test")
    try:
        dn = Deep(1.5, [3, 2])
    except Exception as err:
        print("Error:", err)
    
    print("nx is zero test")
    try:
        dn = Deep(0, [3, 2])
    except Exception as err:
        print("Error:", err)
    
    print("layers as a list with one element test")
    try:
        dn = Deep(5, [3])
        print("L:", dn.L)
    except Exception as err:
        print("Error:", err)
    
    print("layers is not a list test")
    try:
        dn = Deep(5, "not a list")
    except Exception as err:
        print("Error:", err)
    
    print("layers is an empty list test")
    try:
        dn = Deep(5, [])
    except Exception as err:
        print("Error:", err)
    
    print("layers has an element that is not a positive integer test")
    try:
        dn = Deep(5, [3, -2, 4])
    except Exception as err:
        print("Error:", err)

if __name__ == "__main__":
    run_tests()
