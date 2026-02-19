#!/usr/bin/env python3
"""
Module that slices specific columns and rows from a DataFrame.
"""

import pandas as pd


def slice(df):
    """
    Extract High, Low, Close, and Volume_(BTC) columns
    and select every 60th row.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Sliced DataFrame.
    """
    selected = df[["High", "Low", "Close", "Volume_(BTC)"]]

    return selected.iloc[::60]
