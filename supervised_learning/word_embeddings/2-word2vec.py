#!/usr/bin/env python3
"""
Function to create, build, and train a Word2Vec model.
"""

from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    Creates and trains a gensim Word2Vec model.

    Args:
        sentences: list of tokenized sentences
        vector_size: embedding dimensionality
        min_count: minimum word frequency
        window: context window size
        negative: number of negative samples
        cbow: True → CBOW, False → Skip-gram
        epochs: number of training epochs
        seed: random seed
        workers: number of worker threads

    Returns:
        Trained Word2Vec model.
    """
    sg = 0 if cbow else 1

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers,
    )

    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
