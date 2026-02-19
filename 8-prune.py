#!/usr/bin/env python3

"""
Module that removes rows with NaN values in the Close column.
"""

import pandas as pd


def prune(df):
    """
    Remove rows where Close has NaN values.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    return df.dropna(subset=["Close"])
