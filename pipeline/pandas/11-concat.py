#!/usr/bin/env python3
"""
Module that concatenates two DataFrames with hierarchical keys.
"""

import pandas as pd


index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenate two DataFrames after indexing them by Timestamp.

    - Index both DataFrames on Timestamp
    - Include rows from df2 up to and including 1417411920
    - Concatenate df2 on top of df1
    - Add keys 'bitstamp' and 'coinbase'

    Args:
        df1 (pandas.DataFrame): coinbase DataFrame
        df2 (pandas.DataFrame): bitstamp DataFrame

    Returns:
        pandas.DataFrame: Concatenated DataFrame
    """
    # Set Timestamp as index
    df1 = index(df1)
    df2 = index(df2)

    # Select required rows from df2
    df2 = df2.loc[:1417411920]

    # Concatenate with keys
    return pd.concat(
        [df2, df1],
        keys=["bitstamp", "coinbase"]
    )
