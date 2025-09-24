#!/usr/bin/env python3
"""
Main file for task 11
"""

import numpy as np
one_hot = __import__('3-one_hot').one_hot
model = __import__('11-config')

if __name__ == '__main__':
    datasets = np.load('MNIST.npz')
    X_train = datasets['X_train']
    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)

    from 1-input import build_model
    from 2-optimize import optimize_model

    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'], 0.0001, 0.95)
    optimize_model(network, 0.001, 0.9, 0.999)

    # Save and load config
    model.save_config(network, "model_config.json")
    new_network = model.load_config("model_config.json")

    print(type(new_network))
    print(new_network.to_json()[:200], "...")
