#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import random
import os

SEED = 4

# ensure reproducibility
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

dropout_create_layer = __import__('6-dropout_create_layer').dropout_create_layer

# dummy input data: 10 samples, each with 784 features
X = np.random.randint(0, 256, size=(10, 784))

# create a dense layer with dropout
a = dropout_create_layer(X, 256, tf.nn.tanh, 0.8)

# print the output of the first sample
print(a[0])
