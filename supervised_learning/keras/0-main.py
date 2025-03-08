#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
load_model = __import__('9-model').load_model
save_weights = __import__('10-weights').save_weights
load_weights = __import__('10-weights').load_weights

if __name__ == '__main__':
    # Ensure reproducible results
    np.random.seed(0)
    tf.random.set_seed(0)

    # Load the model
    model = load_model('network1.keras')

    # Save the weights to a file
    weights_file = './0-test.keras.weights.h5'
    save_weights(model, weights_file)

    # Output expected success message
    print(f'Weights saved successfully to {weights_file}')

    # Create a new model and load the weights
    model2 = load_model('network1.keras')
    load_weights(model2, weights_file)

    # Output the weights and biases of the first layer for verification
    print(model2.get_weights()[0])
    print(model2.get_weights()[1])
