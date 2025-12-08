#!/usr/bin/env python3
"""Main script to test the Dataset class"""

Dataset = __import__('2-dataset').Dataset

data = Dataset()

for pt, en in data.data_train.take(1):
    print(pt, en)

for pt, en in data.data_valid.take(1):
    print(pt, en)

