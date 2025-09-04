#!/usr/bin/env python3
"""
Creates a confusion matrix from one-hot encoded labels and logits.
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Parameters:
    - labels: one-hot np.ndarray of shape (m, classes), true labels
    - logits: one-hot np.ndarray of shape (m, classes), predicted labels

    Returns:
    - confusion: np.ndarray of shape (classes, classes)
    """
    # Convert one-hot to class indices
    true_classes = np.argmax(labels, axis=1)
    predicted_classes = np.argmax(logits, axis=1)

    num_classes = labels.shape[1]
    confusion = np.zeros((num_classes, num_classes))

    for true, pred in zip(true_classes, predicted_classes):
        confusion[true][pred] += 1

    return confusion
