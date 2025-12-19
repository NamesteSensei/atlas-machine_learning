#!/usr/bin/env python3
"""
Main file to test training of the Transformer model.
"""

import tensorflow as tf

train_transformer = __import__('5-train').train_transformer


if __name__ == "__main__":
    tf.random.set_seed(0)

    transformer = train_transformer(
        N=4,
        dm=128,
        h=8,
        hidden=512,
        max_len=32,
        batch_size=40,
        epochs=2
    )

    print(type(transformer))
