#!/usr/bin/env python3
"""
/** 
 * Module 0-create_placeholders:
 * This module contains a function to create TensorFlow placeholders
 * for the input data and one-hot labels for a neural network.
 */
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()  # /** Disable eager execution for TF v1 compatibility **/

def create_placeholders(nx, classes):
    """
    /** 
     * create_placeholders - Creates placeholders for neural network inputs.
     * @param nx: Number of features (columns) in the input data.
     * @param classes: Number of classes (output nodes) for one-hot labels.
     * @return: Tuple of placeholders (x, y)
     */
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")  # /** Placeholder for input data **/
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")  # /** Placeholder for one-hot labels **/
    return x, y

if __name__ == "__main__":
    # /** Testing the function by printing the placeholders **/
    x, y = create_placeholders(784, 10)
    print(x)
    print(y)
