#!/usr/bin/env python3
"""
Module that rearranges hierarchical indexing and filters timestamps.
"""

import pandas as pd


index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenate two DataFrames with hierarchical indexing.

    - Index both DataFrames on Timestamp
    - Add keys 'bitstamp' and 'coinbase'
    - Rearrange MultiIndex so Timestamp is first level
    - Filter timestamps from 1417411980 to 1417417980 inclusive
    - Ensure chronological order

    Args:
        df1 (pandas.DataFrame): coinbase DataFrame
        df2 (pandas.DataFrame): bitstamp DataFrame

    Returns:
        pandas.DataFrame: Transformed DataFrame
    """
    # Index both DataFrames
    df1 = index(df1)
    df2 = index(df2)

    # Concatenate with keys
    df = pd.concat(
        [df2, df1],
        keys=["bitstamp", "coinbase"]
    )

    # Swap MultiIndex levels
    df = df.swaplevel(0, 1)

    # Sort by Timestamp (chronological order)
    df = df.sort_index()

    # Slice required timestamp range
    return df.loc[1417411980:1417417980]
