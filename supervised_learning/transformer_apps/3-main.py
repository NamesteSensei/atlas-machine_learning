#!/usr/bin/env python3
"""Test script for the Dataset class with batching and filtering"""

import tensorflow as tf
Dataset = __import__('3-dataset').Dataset

tf.random.set_seed(0)

data = Dataset(batch_size=32, max_len=40)

for pt, en in data.data_train.take(1):
    print(pt, en)

for pt, en in data.data_valid.take(1):
    print(pt, en)
