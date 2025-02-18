#!/usr/bin/env python3
"""
Module for calculating the weighted moving average of a dataset.

This implementation includes bias correction for more accurate results.
"""

import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a dataset.

    Parameters:
    data (list): List of numerical values.
    beta (float): Smoothing factor (between 0 and 1).

    Returns:
    list: List of moving averages.
    """
    V = 0  # Initialize moving average
    moving_averages = []

    for t, x in enumerate(data, start=1):
        V = beta * V + (1 - beta) * x  # Compute weighted average
        V_corrected = V / (1 - beta ** t)  # Apply bias correction
        moving_averages.append(V_corrected)

    return moving_averages
