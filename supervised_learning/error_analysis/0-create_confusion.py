#!/usr/bin/env python3
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    Parameters:
    - labels: numpy.ndarray of shape (m, classes),
              true labels (one-hot encoded)
    - logits: numpy.ndarray of shape (m, classes),
              predicted labels (one-hot encoded)

    Returns:
    - confusion: numpy.ndarray of shape (classes, classes) with counts
    """

    # number of data points and classes
    m, classes = labels.shape

    # converts one-hot encoded labels to class indices
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)

    # init an empty confusion matrix
    confusion = np.zeros((classes, classes))

    # populates confusion matrix by counting occurrences
    for i in range(m):
        confusion[true_labels[i], predicted_labels[i]] += 1

    return confusion
