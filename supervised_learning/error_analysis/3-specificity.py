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

    # Get True Positives (TP) - diagonal values
    TP = np.diag(confusion)

    # Get False Positives (FP) - sum of each column minus TP
    FP = np.sum(confusion, axis=0) - TP

    # Get False Negatives (FN) - sum of each row minus TP
    FN = np.sum(confusion, axis=1) - TP

    # Compute True Negatives (TN)
    TN = total - (TP + FP + FN)  # Corrected calculation

    # Compute Specificity
    specificity_values = TN / (TN + FP)

    return specificity_values
