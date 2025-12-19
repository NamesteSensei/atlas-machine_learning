#!/usr/bin/env python3
"""
Task 5 Main File
"""

import tensorflow as tf
from 3-dataset import Dataset
from 5-train import train_transformer


def main():
    """
    Train and test the Transformer model.
    """
    tf.random.set_seed(0)

    dataset = Dataset(
        batch_size=16,
        max_len=40
    )

    transformer = train_transformer(
        dataset=dataset,
        epochs=1
    )

    print("Training completed successfully")


if __name__ == "__main__":
    main()
