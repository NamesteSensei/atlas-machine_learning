#!/usr/bin/env python3
"""Trains a Word2Vec model using Gensim."""

import gensim

def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Trains a Word2Vec model on a list of tokenized sentences.

    Args:
        sentences (list): Tokenized input sentences.
        vector_size (int): Dimensionality of word vectors.
        min_count (int): Minimum word frequency to be considered.
        window (int): Maximum distance between context and target.
        negative (int): Number of negative samples.
        cbow (bool): True for CBOW, False for Skip-gram.
        epochs (int): Number of training iterations.
        seed (int): Seed for reproducibility.
        workers (int): Number of worker threads.

    Returns:
        gensim.models.Word2Vec: The trained model.
    """
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return model
