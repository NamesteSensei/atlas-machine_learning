#!/usr/bin/env python3
"""
Train a Word2Vec model using gensim.
"""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    Trains a gensim Word2Vec model.

    Args:
        sentences: list of tokenized sentences
        vector_size: embedding dimension
        min_count: min word freq
        window: context window size
        negative: negative sampling size
        cbow: True for CBOW, False skip-gram
        epochs: number of training epochs
        seed: random seed
        workers: thread count

    Returns:
        The trained Word2Vec model.
    """

    # Set deterministic training params
    sg_flag = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg_flag,
        negative=negative,
        seed=seed,
        workers=workers
    )

    # Build vocab before training
    model.build_vocab(sentences)

    # Train with deterministic shuffle disabled
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
