import numpy as np
from deep_neural_network import DeepNeuralNetwork

if __name__ == "__main__":
    # Test cases for DeepNeuralNetwork
    try:
        dnn = DeepNeuralNetwork("invalid", [5, 3, 1])
    except Exception as e:
        print(e)  # Expected: nx must be an integer

    try:
        dnn = DeepNeuralNetwork(0, [5, 3, 1])
    except Exception as e:
        print(e)  # Expected: nx must be a positive integer

    try:
        dnn = DeepNeuralNetwork(1.5, [5, 3, 1])
    except Exception as e:
        print(e)  # Expected: nx must be an integer

    try:
        dnn = DeepNeuralNetwork(5, "invalid")
    except Exception as e:
        print(e)  # Expected: layers must be a list of positive integers
    
    try:
        dnn = DeepNeuralNetwork(5, [])
    except Exception as e:
        print(e)  # Expected: layers must be a list of positive integers
    
    try:
        dnn = DeepNeuralNetwork(5, [5, -3, 1])
    except Exception as e:
        print(e)  # Expected: layers must be a list of positive integers
    
    try:
        dnn = DeepNeuralNetwork(5, [5, 3, "invalid"])
    except Exception as e:
        print(e)  # Expected: layers must be a list of positive integers
    
    # Valid case
    np.random.seed(0)
    dnn = DeepNeuralNetwork(5, [5, 3, 1])
    print(dnn.cache)  # Expected: {}
    print(dnn.weights.keys())  # Expected: W1, b1, W2, b2, W3, b3
    print(dnn.L)  # Expected: 3

