#!/usr/bin/env python3

import numpy as np
one_hot = __import__('3-one_hot').one_hot
model = __import__('9-model')
weights = __import__('10-weights')

if __name__ == '__main__':
    # Correctly matching the expected input labels and number of classes
    labels = np.array([
        8, 0, 10, 11, 9, 10, 6, 0, 12, 7, 14, 17, 2, 2, 1, 5, 8, 14, 1, 10,
        7, 11, 1, 15, 16, 5, 17, 14, 0, 0, 9, 5, 7, 5, 14, 1, 17, 1, 10, 7,
        11, 4, 3, 16, 16, 0, 17, 11, 0, 13, 5, 16, 14, 8, 15, 3, 4, 16, 1, 17,
        8, 2, 4, 9, 5, 7, 5, 14, 1, 17, 1, 10, 7, 11, 4, 3, 16, 16, 0, 17, 11,
        0, 13, 5, 16, 14, 8, 15, 3, 4, 16, 1, 17, 8, 2, 4, 9, 5, 7
    ])
    classes = 18  # Total number of unique classes
    one_hot_matrix = one_hot(labels, classes)
    print(one_hot_matrix)

    # Load the initial model
    network = model.load_model('network1.keras')

    # Display weights before saving
    print("Initial model weights:")
    print(network.get_weights())

    # Save weights to a file
    weights.save_weights(network, './0-test.keras.weights.h5')
    print("Weights saved successfully to ./0-test.keras.weights.h5")

    # Load a new instance of the model
    network2 = model.load_model('network1.keras')

    # Display weights before loading
    print("New model weights before loading saved weights:")
    print(network2.get_weights())

    # Load the weights from the saved file
    weights.load_weights(network2, './0-test.keras.weights.h5')
    print("Weights loaded successfully from ./0-test.keras.weights.h5")

    # Display weights after loading
    print("New model weights after loading saved weights:")
    print(network2.get_weights())
