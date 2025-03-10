#!/usr/bin/env python3

"""
Main script for testing a Keras neural network model.

This script loads the MNIST test dataset, reshapes the input data,
and evaluates a pre-trained model using the test data.
It prints the loss and accuracy of the model on the test data.
"""

import numpy as np
one_hot = __import__('3-one_hot').one_hot
load_model = __import__('9-model').load_model
test_model = __import__('12-test').test_model

if __name__ == '__main__':
    # Load the MNIST dataset
    datasets = np.load('MNIST.npz')
    X_test = datasets['X_test']
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = datasets['Y_test']
    
    # Convert labels to one-hot encoding
    Y_test_oh = one_hot(Y_test)

    # Load the pre-trained model
    network = load_model('network2.keras')

    # Evaluate the model using the test data and labels
    eval = test_model(network, X_test, Y_test_oh)
    
    # Print the evaluation metrics
    print(eval)
    print("Loss:", np.round(eval[0], 3))
    print("Accuracy:", np.round(eval[1], 3))
