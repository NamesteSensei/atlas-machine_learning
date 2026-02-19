#!/usr/bin/env python3
"""
Module that renames and formats a DataFrame.
"""

import pandas as pd


def rename(df):
    """
    Rename Timestamp column to Datetime,
    convert it to datetime format,
    and return only Datetime and Close columns.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Modified DataFrame.
    """
    # Rename column
    df = df.rename(columns={"Timestamp": "Datetime"})

    # Convert Unix timestamp to datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")

    # Keep only required columns
    return df[["Datetime", "Close"]]
