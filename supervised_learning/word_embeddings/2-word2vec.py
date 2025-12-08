#!/usr/bin/env python3
"""Module that builds and trains a Word2Vec model using Gensim."""

from gensim.models import Word2Vec

def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Word2Vec model using Gensim.

    Args:
        sentences (list): List of tokenized sentences (list of list of str).
        vector_size (int): Dimensionality of embedding vectors.
        min_count (int): Minimum word frequency to be included in the model.
        window (int): Maximum distance between current and predicted word.
        negative (int): Size of negative sampling.
        cbow (bool): True for CBOW; False for Skip-gram.
        epochs (int): Number of training iterations.
        seed (int): Random seed.
        workers (int): Number of worker threads.

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    sg = 0 if cbow else 1

    model = Word2Vec(sentences=sentences,
                     vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     workers=workers,
                     seed=seed,
                     sg=sg,
                     negative=negative)

    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model
