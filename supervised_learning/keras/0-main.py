#!/usr/bin/env python3
"""Test script for 0-sequential.py"""

import numpy as np
import tensorflow as tf
from 0-sequential import build_model

if __name__ == '__main__':
    np.random.seed(0)
    tf.random.set_seed(0)

    model = build_model(784, [256, 128, 10],
                        ['relu', 'relu', 'softmax'],
                        0.01, 0.8)
    model.summary()
