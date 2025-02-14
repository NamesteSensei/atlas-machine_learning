#!/usr/bin/env python3
"""Build, train, and save a neural network classifier."""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Args:
        X_train: numpy.ndarray of shape (m, nx) containing the training input
                 data.
        Y_train: numpy.ndarray of shape (m, classes) containing the training
                 labels.
        X_valid: numpy.ndarray of shape (m, nx) containing the validation input
                 data.
        Y_valid: numpy.ndarray of shape (m, classes) containing the validation
                 labels.
        layer_sizes: list containing the number of nodes in each layer.
        activations: list of activation functions for each layer.
        alpha: learning rate.
        iterations: number of iterations to train over.
        save_path: path where the model should be saved.

    Returns:
        The path where the model was saved.
    """
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    x, y = create_placeholders(nx, classes)

    # Build the network and obtain the predictions
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Create the training operation
    train_op = create_train_op(loss, alpha)

    # Add tensors and operation to the collection
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            if i % 100 == 0 or i == iterations:
                # Evaluate training metrics
                train_cost, train_acc = sess.run(
                    [loss, accuracy],
                    feed_dict={x: X_train, y: Y_train}
                )
                # Evaluate validation metrics
                valid_cost, valid_acc = sess.run(
                    [loss, accuracy],
                    feed_dict={x: X_valid, y: Y_valid}
                )
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_acc))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_acc))
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        saver.save(sess, save_path)
    return save_path
