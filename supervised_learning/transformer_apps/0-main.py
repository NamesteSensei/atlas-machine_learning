#!/usr/bin/env python3
"""
Main file for testing Dataset class.
"""

from pprint import pprint
from 0-dataset import Dataset

# Create a Dataset instance
data = Dataset()

# Display a few examples from the dataset
for pt, en in data.data_train.take(2):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))

# Show the tokenizer classes
print(type(data.tokenizer_pt))
print(type(data.tokenizer_en))

