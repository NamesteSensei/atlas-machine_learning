#!/usr/bin/env python3
"""
Calculates the F1 score for each class in a confusion matrix.
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    Parameters:
        confusion (numpy.ndarray): shape (classes, classes) where rows
            represent the true labels and columns represent the predicted
            labels.

    Returns:
        numpy.ndarray: shape (classes,) containing the F1 score of each class.
    """
    recall = sensitivity(confusion)
    prec = precision(confusion)

    return 2 * (prec * recall) / (prec + recall)
