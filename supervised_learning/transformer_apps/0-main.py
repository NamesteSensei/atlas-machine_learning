#!/usr/bin/env python3

Dataset = __import__('3-dataset').Dataset
import tensorflow as tf

tf.random.set_seed(0)
data = Dataset(32, 40)

for pt, en in data.data_train.take(1):
    print("Train: pt sentence shape is valid.")
    print("Train: en sentence shape is valid.")

for pt, en in data.data_valid.take(1):
    print("Validation: pt sentence shape is valid.")
    print("Validation: en sentence shape is valid.")
