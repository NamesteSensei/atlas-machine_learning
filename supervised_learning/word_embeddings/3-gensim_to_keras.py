#!/usr/bin/env python3
"""
Convert a gensim Word2Vec model into a Keras Embedding layer.
"""

import numpy as np
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """
    Converts a gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model: Trained gensim Word2Vec model.

    Returns:
        keras.layers.Embedding: a trainable Keras Embedding layer
        initialized with the gensim weights.
    """
    # Weight matrix from gensim (vocab_size x vector_size)
    w2v_weights = model.wv.vectors

    vocab_size, vector_size = w2v_weights.shape

    # Keras requires index 0 to be reserved → shift weights by one row
    embedding_matrix = np.zeros((vocab_size + 1, vector_size))
    embedding_matrix[1:] = w2v_weights

    # Build Keras Embedding layer
    embedding_layer = Embedding(
        input_dim=vocab_size + 1,
        output_dim=vector_size,
        weights=[embedding_matrix],
        trainable=True
    )

    return embedding_layer
