#!/usr/bin/env python3
import tensorflow as tf

Dataset = __import__('2-dataset').Dataset
data = Dataset()

for pt, en in data.data_train.take(1):
    print(pt.dtype == tf.int64)
    print(en.dtype == tf.int64)
    print(pt.dtype)
    print(en.dtype)

for pt, en in data.data_valid.take(1):
    print(pt.dtype == tf.int64)
    print(en.dtype == tf.int64)
    print(pt.dtype)
    print(en.dtype)
