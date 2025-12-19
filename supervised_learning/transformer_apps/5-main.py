#!/usr/bin/env python3
"""
Task 5 Main File
"""

import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
train_transformer = __import__('5-train').train_transformer


def main():
    """
    Train and test the Transformer model.
    """
    tf.random.set_seed(0)

    dataset = Dataset(batch_size=16, max_len=40)

    transformer = train_transformer(dataset=dataset, epochs=1)

    print(type(transformer))


if __name__ == "__main__":
    main()
