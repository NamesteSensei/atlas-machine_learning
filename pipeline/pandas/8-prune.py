#!/usr/bin/env python3
"""
Module that removes rows where Close contains NaN values.
"""


def prune(df):
    """
    Remove rows where the Close column has NaN values.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    return df.dropna(subset=["Close"])
