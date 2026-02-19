#!/usr/bin/env python3
"""
Module that converts selected DataFrame values to a NumPy array.
"""

import pandas as pd


def array(df):
    """
    Select the last 10 rows of High and Close columns
    and convert them to a NumPy array.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        numpy.ndarray: Selected values as NumPy array.
    """
    # Select last 10 rows and required columns
    selected = df[["High", "Close"]].tail(10)

    # Convert to NumPy array
    return selected.to_numpy()
