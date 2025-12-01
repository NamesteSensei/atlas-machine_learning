#!/usr/bin/env python3
"""
Main script to test the Dataset class.
"""

import importlib.util
import sys

# Dynamically load 0-dataset.py
spec = importlib.util.spec_from_file_location("dataset", "0-dataset.py")
dataset = importlib.util.module_from_spec(spec)
sys.modules["dataset"] = dataset
spec.loader.exec_module(dataset)

Dataset = dataset.Dataset


if __name__ == "__main__":
    data = Dataset()

    print("\nSample training data:")
    for pt, en in data.data_train.take(1):
        print("PT:", pt.numpy().decode("utf-8"))
        print("EN:", en.numpy().decode("utf-8"))

    print("\nSample validation data:")
    for pt, en in data.data_valid.take(1):
        print("PT:", pt.numpy().decode("utf-8"))
        print("EN:", en.numpy().decode("utf-8"))

    print("\nTokenizer objects:")
    print(type(data.tokenizer_pt))
    print(type(data.tokenizer_en))
