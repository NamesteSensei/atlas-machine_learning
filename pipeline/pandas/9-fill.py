#!/usr/bin/env python3
"""
Module that fills missing values in a DataFrame
according to specified rules.
"""

import pandas as pd


def fill(df):
    """
    Clean and fill missing values in the DataFrame.

    - Remove Weighted_Price column
    - Forward fill Close column
    - Fill High, Low, Open with corresponding Close value
    - Fill Volume_(BTC) and Volume_(Currency) with 0

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Modified DataFrame.
    """
    # Remove Weighted_Price column
    df = df.drop(columns=["Weighted_Price"])

    # Forward fill Close column
    df["Close"] = df["Close"].ffill()

    # Fill High, Low, Open with Close value
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])
    df["Open"] = df["Open"].fillna(df["Close"])

    # Fill Volume columns with 0
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
