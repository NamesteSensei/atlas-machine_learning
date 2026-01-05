#!/usr/bin/env python3

import tensorflow as tf
train_transformer = __import__('5-train').train_transformer

tf.random.set_seed(0)

model = train_transformer(
    4,       # N (layers)
    128,     # dm (depth)
    8,       # heads
    512,     # hidden size
    32,      # max_len
    40,      # batch size
    20       # training for 20 epochs
)
print(type(model))
