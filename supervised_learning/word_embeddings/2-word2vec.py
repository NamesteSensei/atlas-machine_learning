#!/usr/bin/env python3
"""Trains a Word2Vec model using Gensim."""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Trains a Word2Vec model using only gensim.

    Args:
        sentences (list): Tokenized input sentences.
        vector_size (int): Embedding size.
        min_count (int): Minimum word frequency.
        window (int): Context window size.
        negative (int): Negative sampling size.
        cbow (bool): True for CBOW, False for Skip-gram.
        epochs (int): Number of training epochs.
        seed (int): Random seed for reproducibility.
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
