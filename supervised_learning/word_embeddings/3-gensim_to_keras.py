#!/usr/bin/env python3
"""
Convert a gensim Word2Vec model into a Keras Embedding layer.
"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model: A trained gensim Word2Vec model.

    Returns:
        A trainable tf.keras.layers.Embedding layer initialized with
        the gensim word vectors.
    """
    # Extract the gensim weight matrix (vocab_size x vector_size)
    w2v_weights = model.wv.vectors
    vocab_size, vector_size = w2v_weights.shape

    # Build TensorFlow-compatible embedding matrix
    # Reserve index 0 → pad with an all-zero row
    embedding_matrix = tf.concat(
        [
            tf.zeros((1, vector_size), dtype=tf.float32),
            tf.constant(w2v_weights, dtype=tf.float32)
        ],
        axis=0
    )

    # Create the Keras Embedding layer
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size + 1,
        output_dim=vector_size,
        embeddings_initializer=tf.keras.initializers.Constant(
            embedding_matrix
        ),
        trainable=True
    )

    return embedding_layer
