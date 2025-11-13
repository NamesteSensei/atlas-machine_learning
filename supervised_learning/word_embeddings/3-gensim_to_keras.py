#!/usr/bin/env python3
"""
Convert a gensim Word2Vec model to a trainable Keras Embedding layer.
"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a trained gensim Word2Vec model into a Keras Embedding layer.

    Args:
        model: a trained gensim Word2Vec instance.

    Returns:
        A tf.keras.layers.Embedding layer with the model's weights. The layer
        is trainable and uses the vocabulary index mapping from gensim.
    """
    vocab_size = len(model.wv)
    emb_dim = model.vector_size

    weights = model.wv.vectors

    layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=emb_dim,
        weights=[weights],
        trainable=True
    )
    return layer
