#!/usr/bin/env python3
"""
This module calculates the sensitivity (recall) for each class
in a confusion matrix.
"""

import numpy as np


def sensitivity(confusion):
    """
    Computes sensitivity (recall) for each class in a confusion matrix.

    Parameters:
    - confusion: numpy.ndarray of shape (classes, classes)
                 where rows represent actual labels
                 and columns represent predicted labels.

    Returns:
    - numpy.ndarray of shape (classes,)
      containing sensitivity for each class.
    """

    # Get the True Positives (TP) - diagonal values
    TP = np.diag(confusion)

    # Get the sum of each row (TP + FN)
    FN = np.sum(confusion, axis=1)

    # Compute Sensitivity (Recall)
    sensitivity_values = TP / FN

    return sensitivity_values
