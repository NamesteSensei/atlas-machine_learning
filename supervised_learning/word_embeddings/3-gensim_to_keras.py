#!/usr/bin/env python3
"""Convert gensim Word2Vec to Keras Embedding"""

from tensorflow.keras.layers import Embedding
import numpy as np

def gensim_to_keras(model):
    """
    Converts a trained gensim Word2Vec model into
    a Keras Embedding layer.

    Parameters:
    - model: trained gensim Word2Vec model

    Returns:
    - Keras Embedding layer
    """

    # Extract weights matrix
    weights = model.wv.vectors
    vocab_size, embedding_dim = weights.shape

    # Create a Keras embedding layer with weights
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True  # You can continue training in Keras
    )

    return embedding_layer
