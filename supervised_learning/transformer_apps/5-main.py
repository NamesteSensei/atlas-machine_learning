#!/usr/bin/env python3
"""
Main test script for Task 5: Training the Transformer
"""
import tensorflow as tf
train_transformer = __import__('5-train').train_transformer


def test_train():
    """
    Sets the random seed and runs the training function
    """
    tf.random.set_seed(0)
    # Parameters: N, dm, h, hidden, max_len, batch_size, epochs
    transformer = train_transformer(4, 128, 8, 512, 32, 40, 2)
    print(type(transformer))


if __name__ == "__main__":
    test_train()
