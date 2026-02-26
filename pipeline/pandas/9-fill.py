#!/usr/bin/env python3
"""
Module that fills missing values in a DataFrame.
"""


def fill(df):
    """
    Clean and fill missing values in the DataFrame.

    - Remove Weighted_Price column
    - Forward-fill Close values
    - Fill High, Low, Open with corresponding Close value
    - Fill Volume columns with 0

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    df = df.drop(columns=["Weighted_Price"])

    df["Close"] = df["Close"].ffill()

    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])
    df["Open"] = df["Open"].fillna(df["Close"])

    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
