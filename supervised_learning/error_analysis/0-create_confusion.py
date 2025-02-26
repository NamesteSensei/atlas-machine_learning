#!/usr/bin/env python3
"""
This module creates a confusion matrix for classification models.
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Builds a confusion matrix from predicted and true labels.

    Parameters:
    - labels: numpy.ndarray of shape (m, classes)
              One-hot encoded true labels.
    - logits: numpy.ndarray of shape (m, classes)
              One-hot encoded predicted labels.

    Returns:
    - confusion: numpy.ndarray of shape (classes, classes)
                 Counts of correct and incorrect predictions.
    """

    # Get the number of classes
    m, classes = labels.shape

    # Convert one-hot encoded labels to class indices
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)

    # Initialize an empty confusion matrix
    confusion = np.zeros((classes, classes))

    # Count occurrences for each prediction
    for i in range(m):
        confusion[true_labels[i], predicted_labels[i]] += 1

    return confusion
