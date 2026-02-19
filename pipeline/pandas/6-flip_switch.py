#!/usr/bin/env python3
"""
Module that sorts a DataFrame in reverse chronological order
and transposes it.
"""

import pandas as pd


def flip_switch(df):
    """
    Sort DataFrame in reverse chronological order
    and transpose it.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Transformed DataFrame.
    """
    # Sort by Timestamp descending
    sorted_df = df.sort_values(by="Timestamp", ascending=False)

    # Transpose the sorted DataFrame
    return sorted_df.transpose()
