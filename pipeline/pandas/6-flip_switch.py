#!/usr/bin/env python3
"""
Module that sorts a DataFrame in reverse chronological order
and transposes it.
"""


def flip_switch(df):
    """
    Sort DataFrame in reverse chronological order
    and transpose it.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Transformed DataFrame.
    """
    sorted_df = df.sort_values(by="Timestamp", ascending=False)
    return sorted_df.transpose()
