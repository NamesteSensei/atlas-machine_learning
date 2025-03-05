#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import os
import random

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

l2_reg_create_layer = __import__('3-l2_reg_create_layer').l2_reg_create_layer

X = np.random.randint(0, 256, size=(10, 784)).astype(np.float32)
layer_output = l2_reg_create_layer(X, 256, tf.nn.tanh, 0.05)

print(f"Layer Output Shape: {layer_output.shape}")
print(f"Layer Output (First Row): {layer_output[0]}")
