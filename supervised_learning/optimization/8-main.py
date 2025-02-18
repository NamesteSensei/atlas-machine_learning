#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os

SEED = 0

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

create_RMSProp_op = __import__('8-RMSProp').create_RMSProp_op


def one_hot(Y, classes):
    """Convert an array to a one-hot matrix."""
    one_hot = np.zeros((Y.shape[0], classes))
