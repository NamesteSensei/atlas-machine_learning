#!/usr/bin/env python3
"""
Main script for making predictions using a trained Keras model.

Loads the MNIST test dataset, reshapes the input data, makes predictions
using a pre-trained model, and compares the predicted classes to the actual
labels.
"""

import numpy as np
one_hot = __import__('3-one_hot').one_hot
load_model = __import__('9-model').load_model
predict = __import__('13-predict').predict

if __name__ == '__main__':
    # Load and prepare the MNIST dataset
    datasets = np.load('MNIST.npz')
    X_test = datasets['X_test'].reshape(datasets['X_test'].shape[0], -1)
    Y_test = datasets['Y_test']

    # Load the pre-trained model
    network = load_model('network2.keras')

    # Make predictions using the model
    Y_pred = predict(network, X_test)

    # Output the predictions and comparison with actual labels
    print(Y_pred)
    print(np.argmax(Y_pred, axis=1))
    print(Y_test)
