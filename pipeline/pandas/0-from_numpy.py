#!/usr/bin/env python3
"""
Module that converts a NumPy ndarray into a Pandas DataFrame
with alphabetically labeled columns.
"""

import pandas as pd


def from_numpy(array):
    """
    Convert a NumPy ndarray into a Pandas DataFrame.

    Columns are labeled alphabetically in uppercase starting
    from 'A'. Maximum of 26 columns.

    Args:
        array (numpy.ndarray): Input NumPy array.

    Returns:
        pandas.DataFrame: DataFrame with labeled columns.
    """
    num_cols = array.shape[1]

    columns = [chr(ord('A') + i) for i in range(num_cols)]

    return pd.DataFrame(array, columns=columns)
