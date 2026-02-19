#!/usr/bin/env python3


"""
Module that computes descriptive statistics for a DataFrame.
"""

import pandas as pd


def analyze(df):
    """
    Compute descriptive statistics for all columns
    except the Timestamp column.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame of descriptive statistics.
    """
    df = df.drop(columns=["Timestamp"])
    return df.describe()
