#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np


def main():
    """
    Main function to test the Neuron class from 6-neuron.py.
    Loads training and development data, trains the neuron, evaluates
    performance, and displays sample predictions.
    """
    # Import the Neuron class from 6-neuron.py.
    Neuron = __import__('6-neuron').Neuron

    # Load training data.
    lib_train = np.load('../data/train.npz')
    X_train_3D, Y_train = lib_train['X'], lib_train['Y']
    # Reshape training data: each column is a flattened example.
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

    # Load development (dev) data.
    lib_dev = np.load('../data/dev.npz')
    X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
    # Reshape dev data: each column is a flattened example.
    X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

    # Set random seed for reproducibility.
    np.random.seed(0)

    # Instantiate the neuron using the number of input features.
    neuron = Neuron(X_train.shape[0])

    # Train the neuron for a set number of iterations (e.g., 10 for demonstration)
    A_train, cost_train = neuron.train(X_train, Y_train, iterations=10, alpha=0.05)

    # Calculate training accuracy.
    accuracy_train = np.sum(A_train == Y_train) / Y_train.shape[1] * 100

    # Print training results.
    print("Train cost:", np.round(cost_train, decimals=10))
    print("Train accuracy: {}%".format(np.round(accuracy_train, decimals=10)))
    print("Train predictions:", A_train)
    print("Neuron activated output:", np.round(neuron.A, decimals=10))

    # Evaluate the neuron on the development data.
    A_dev, cost_dev = neuron.evaluate(X_dev, Y_dev)
    accuracy_dev = np.sum(A_dev == Y_dev) / Y_dev.shape[1] * 100

    # Print development results.
    print("Dev cost:", np.round(cost_dev, decimals=10))
    print("Dev accuracy: {}%".format(np.round(accuracy_dev, decimals=10)))
    print("Dev predictions:", A_dev)
    print("Neuron activated output on dev:", np.round(neuron.A, decimals=10))

    # Visualize the first 100 images from the development set with their predicted labels.
    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        ax = fig.add_subplot(10, 10, i + 1)
        ax.imshow(X_dev_3D[i])
        ax.set_title(str(A_dev[0, i]))
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
