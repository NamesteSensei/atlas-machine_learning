#!/usr/bin/env python3
"""
Calculates sensitivity (recall) for each class in a confusion matrix.
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Parameters:
    - confusion: numpy.ndarray of shape (classes, classes)
        where rows = true labels, columns = predicted labels

    Returns:
    - numpy.ndarray of shape (classes,) containing sensitivity of each class
    """
    true_positives = np.diag(confusion)
    false_negatives = np.sum(confusion, axis=1) - true_positives
    sensitivity = true_positives / (true_positives + false_negatives)
    return sensitivity
