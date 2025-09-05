#!/usr/bin/env python3
"""
Calculates specificity for each class in a confusion matrix.
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Parameters:
    - confusion: numpy.ndarray of shape (classes, classes)
        where rows = true labels, columns = predicted labels

    Returns:
    - numpy.ndarray of shape (classes,) containing specificity of each class
    """
    total = np.sum(confusion)
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    false_negatives = np.sum(confusion, axis=1) - true_positives
    true_negatives = total - (true_positives + false_positives + false_negatives)

    specificity = true_negatives / (true_negatives + false_positives)
    return specificity
