#!/usr/bin/env python3
"""Convert a gensim Word2Vec model to a keras Embedding layer"""

import gensim
from tensorflow import keras


def gensim_to_keras(model):
    """
    Convert a trained gensim Word2Vec model to a keras Embedding.

    Args:
        model: trained gensim Word2Vec model

    Returns:
        keras.layers.Embedding instance (trainable)
    """
    # extract vocabulary size and vector dimensions
    vocab_size = len(model.wv)
    vector_size = model.vector_size

    # get gensim embedding weights
    weights = model.wv.vectors

    # create keras embedding layer
    embedding = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True
    )

    return embedding
