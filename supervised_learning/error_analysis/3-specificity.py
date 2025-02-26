#!/usr/bin/env python3
"""
This module calculates the specificity for each class
in a confusion matrix.
"""

import numpy as np


def specificity(confusion):
    """
    Computes specificity for each class in a confusion matrix.

    Parameters:
    - confusion: numpy.ndarray of shape (classes, classes)
                 where rows represent actual labels
                 and columns represent predicted labels.

    Returns:
    - numpy.ndarray of shape (classes,)
      containing specificity for each class.
    """

    # Get the total number of samples
    total = np.sum(confusion)

    # Get the True Positives (TP) - diagonal values
    TP = np.diag(confusion)

    # Get the sum of each row (TP + FN) - actual instances
    FN = np.sum(confusion, axis=1)

    # Get the sum of each column (TP + FP) - predicted instances
    FP = np.sum(confusion, axis=0)

    # Compute True Negatives (TN)
    TN = total - (FN + FP - TP)

    # Compute Specificity
    specificity_values = TN / (TN + FP)

    return specificity_values
