#!/usr/bin/env python3
"""
Module that cleans, aggregates, and visualizes Bitcoin data.
"""

import matplotlib.pyplot as plt
import pandas as pd


def visualize(df):
    """
    Clean, transform, and aggregate the DataFrame,
    then plot daily data from 2017 onward.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Transformed daily DataFrame.
    """
    # Remove Weighted_Price column
    df = df.drop(columns=["Weighted_Price"])

    # Rename Timestamp to Date
    df = df.rename(columns={"Timestamp": "Date"})

    # Convert timestamp to datetime
    df["Date"] = pd.to_datetime(df["Date"], unit="s")

    # Set Date as index
    df = df.set_index("Date")

    # Fill missing Close with previous value
    df["Close"] = df["Close"].ffill()

    # Fill High, Low, Open with corresponding Close
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])
    df["Open"] = df["Open"].fillna(df["Close"])

    # Fill volume columns with 0
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    # Filter from 2017 onward
    df = df[df.index >= "2017-01-01"]

    # Daily aggregation
    df_daily = df.resample("D").agg({
        "High": "max",
        "Low": "min",
        "Open": "mean",
        "Close": "mean",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum"
    })

    # Plot Close price
    df_daily["Close"].plot(figsize=(10, 5))
    plt.title("Bitcoin Daily Close Price (2017+)")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.tight_layout()
    plt.show()

    return df_daily
