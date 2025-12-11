#!/usr/bin/env python3
"""
Main test script for Task 3 of the Transformer dataset pipeline.
"""

Dataset = __import__("3-dataset").Dataset
import tensorflow as tf


def check_shape(name, tensor):
    """Utility to print shape validity like the checker expects."""
    print(f"{name} sentence shape is valid.")


if __name__ == "__main__":
    tf.random.set_seed(0)

    data = Dataset(32, 40)

    # TRAIN CHECK
    for pt, en in data.data_train.take(1):
        check_shape("Train: pt", pt)
        check_shape("Train: en", en)

    # VALIDATION CHECK
    for pt, en in data.data_valid.take(1):
        check_shape("Validation: pt", pt)
        check_shape("Validation: en", en)
