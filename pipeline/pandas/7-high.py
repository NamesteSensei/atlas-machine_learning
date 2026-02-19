#!/usr/bin/env python3
"""
Module that sorts a DataFrame by High price in descending order.
"""

import pandas as pd


def high(df):
    """
    Sort the DataFrame by the High column in descending order.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Sorted DataFrame.
    """
    return df.sort_values(by="High", ascending=False)
