#!/usr/bin/env python3
import tensorflow as tf

Dataset = __import__('0-dataset').Dataset
data = Dataset()

for pt, en in data.data_train.take(1):
    print(type(pt) is tf.Tensor)
    print(type(en) is tf.Tensor)
    print(pt.dtype)
    print(en.dtype)

for pt, en in data.data_valid.take(1):
    print(type(pt) is tf.Tensor)
    print(type(en) is tf.Tensor)
    print(pt.dtype)
    print(en.dtype)

