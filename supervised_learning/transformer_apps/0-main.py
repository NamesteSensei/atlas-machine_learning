#!/usr/bin/env python3
"""Test for Dataset class"""

import tensorflow as tf

Dataset = __import__('2-dataset').Dataset

data = Dataset()
for pt, en in data.data_train.take(1):
    print(isinstance(pt, tf.Tensor))
    print(isinstance(en, tf.Tensor))
    print(pt.dtype)
    print(en.dtype)
for pt, en in data.data_valid.take(1):
    print(isinstance(pt, tf.Tensor))
    print(pt.dtype)
    print(en.dtype)

