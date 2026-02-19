#!/usr/bin/env python3
"""
Module that creates a Pandas DataFrame from a dictionary.
"""

import pandas as pd


# Dictionary data
data = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
}

# Row labels
index_labels = ["A", "B", "C", "D"]

# Create DataFrame
df = pd.DataFrame(data, index=index_labels)
