#!/usr/bin/env python3

"""
Mole that slices specific columns of a DataFrame.
"""


def slice(df):
    """
    Extract specific columns and select every 60th row.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Sliced DataFrame.
    """
    selected = df[["High", "Low", "Close", "Volume_(BTC)"]]
    return selected.iloc[::60]
