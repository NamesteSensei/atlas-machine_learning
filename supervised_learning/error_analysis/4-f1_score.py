#!/usr/bin/env python3
"""
This module calculates the F1 score for each class
in a confusion matrix.
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Computes the F1 score for each class in a confusion matrix.

    Parameters:
    - confusion: numpy.ndarray of shape (classes, classes)
                 where rows represent actual labels
                 and columns represent predicted labels.

    Returns:
    - numpy.ndarray of shape (classes,)
      containing the F1 score for each class.
    """

    # Compute Sensitivity and Precision using imported functions
    recall = sensitivity(confusion)
    prec = precision(confusion)

    # Compute F1 Score using the formula
    f1 = 2 * (prec * recall) / (prec + recall)

    return f1
