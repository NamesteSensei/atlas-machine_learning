#!/usr/bin/env python3
"""
Main file to test loading the TED HRLR translation dataset using dynamic import.

This script dynamically imports the `0-dataset.py` module so we can access
the `load_dataset` function despite the filename beginning with a digit,
which Python normally cannot import directly.

Allowed imports:
    - tensorflow
    - tensorflow_datasets as tfds
"""

import tensorflow as tf  # noqa: F401
import tensorflow_datasets as tfds  # noqa: F401


def dynamic_import(module_name):
    """
    Dynamically import a module by name.

    Args:
        module_name (str): Name of the module (e.g., "0-dataset").

    Returns:
        module: The imported module object.
    """
    return __import__(module_name.replace("-", "_"))


if __name__ == "__main__":
    dataset_module = __import__("0-dataset")
    ds = dataset_module.load_dataset()

    for pt, en in ds.take(1):
        print("Portuguese:", pt.numpy().decode("utf-8"))
        print("English:   ", en.numpy().decode("utf-8"))
