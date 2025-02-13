#!/usr/bin/env python3

import numpy as np
from deep_neural_network import DeepNeuralNetwork

np.random.seed(0)
X = np.random.randn(10, 5)
Y = np.array([[1, 0, 1, 0, 1]])

dnn = DeepNeuralNetwork(10, [5, 5, 5, 5, 5])
A, cache = dnn.forward_prop(X)
print("A:", A)

print("\nCost:", dnn.cost(Y, A))

prediction, cost = dnn.evaluate(X, Y)
print("\nPrediction:", prediction)
print("Evaluation Cost:", cost)

dnn.train(X, Y, iterations=100, alpha=0.01)
