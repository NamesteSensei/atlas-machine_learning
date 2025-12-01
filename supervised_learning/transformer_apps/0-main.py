#!/usr/bin/env python3
"""
Main script to test the Dataset class.
Uses dynamic import because the project file is named `0-dataset.py`.
"""

import importlib.util
import sys

# Dynamically load the module 0-dataset.py
spec = importlib.util.spec_from_file_location("dataset", "0-dataset.py")
dataset = importlib.util.module_from_spec(spec)
sys.modules["dataset"] = dataset
spec.loader.exec_module(dataset)

# Import the Dataset class dynamically
Dataset = dataset.Dataset


if __name__ == "__main__":
    data = Dataset()

    print("\n🔹 Sample training data:")
    for pt, en in data.data_train.take(1):
        print("PT:", pt.numpy().decode("utf-8"))
        print("EN:", en.numpy().decode("utf-8"))

    print("\n🔹 Sample validation data:")
    for pt, en in data.data_valid.take(1):
        print("PT:", pt.numpy().decode("utf-8"))
        print("EN:", en.numpy().decode("utf-8"))

    print("\n🔹 Tokenizer types:")
    print("Portuguese Tokenizer:", type(data.tokenizer_pt))
    print("English Tokenizer:", type(data.tokenizer_en))
