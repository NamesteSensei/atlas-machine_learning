#!/usr/bin/env python3
"""
Module that sets the Timestamp column as the index.
"""

import pandas as pd


def index(df):
    """
    Set the Timestamp column as the DataFrame index.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Modified DataFrame with Timestamp as index.
    """
    return df.set_index("Timestamp")
