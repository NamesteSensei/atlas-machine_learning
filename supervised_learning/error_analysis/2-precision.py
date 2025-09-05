#!/usr/bin/env python3
"""
Calculates precision for each class in a confusion matrix.
"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Parameters:
    - confusion: numpy.ndarray of shape (classes, classes)
        where rows = true labels, columns = predicted labels

    Returns:
    - numpy.ndarray of shape (classes,) containing precision of each class
    """
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    precision = true_positives / (true_positives + false_positives)
    return precision
