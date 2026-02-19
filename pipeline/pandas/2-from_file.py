#!/usr/bin/env python3
"""
Module that loads data from a file into a Pandas DataFrame.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Load data from a file into a Pandas DataFrame.

    Args:
        filename (str): Path to the file.
        delimiter (str): Column separator.

    Returns:
        pandas.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(filename, sep=delimiter)
