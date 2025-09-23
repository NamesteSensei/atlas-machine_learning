#!/usr/bin/env python3
"""
1-main.py
---------
Test file for Task 1: Functional API model builder.
Ensures reproducibility with fixed seeds.
"""

# Force Seed - fix for Keras reproducibility
SEED = 8

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

# Import the build_model function from 1-input.py
build_model = __import__('1-input').build_model

if __name__ == '__main__':
    # Build model: input=784 features, 2 hidden layers of 256, output=10 classes
    network = build_model(
        784,
        [256, 256, 10],
        ['tanh', 'tanh', 'softmax'],
        0.001,
        0.95
    )

    # Print summary of the architecture
    network.summary()

    # Print L2 regularization losses
    print(network.losses)
