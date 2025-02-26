#!/usr/bin/env python3
"""
This module calculates the precision for each class
in a confusion matrix.
"""

import numpy as np


def precision(confusion):
    """
    Computes precision for each class in a confusion matrix.

    Parameters:
    - confusion: numpy.ndarray of shape (classes, classes)
                 where rows represent actual labels
                 and columns represent predicted labels.

    Returns:
    - numpy.ndarray of shape (classes,)
      containing precision for each class.
    """

    # Get the True Positives (TP) - diagonal values
    TP = np.diag(confusion)

    # Get the sum of each column (TP + FP)
    FP = np.sum(confusion, axis=0)

    # Compute Precision
    precision_values = TP / FP

    return precision_values
