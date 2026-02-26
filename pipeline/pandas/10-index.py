#!/usr/bin/env python3
"""
Module that sets Timestamp as the index of a DataFrame.
"""


def index(df):
    """
    Set the Timestamp column as the index.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame indexed by Timestamp.
    """
    return df.set_index("Timestamp")
