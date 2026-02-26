#!/usr/bin/env python3
"""
Module that computes descriptive statistics of a DataFrame.
"""


def analyze(df):
    """
    Compute descriptive statistics for all columns except Timestamp.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Descriptive statistics.
    """
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    return df.describe()
