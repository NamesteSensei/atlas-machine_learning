#!/usr/bin/env python3
"""Train a Word2Vec model using gensim"""

import gensim


def word2vec_model(
        sentences,
        vector_size=100,
        min_count=5,
        window=5,
        negative=5,
        cbow=True,
        epochs=5,
        seed=0,
        workers=1):
    """
    Train a Word2Vec model on tokenized text data.

    Args:
        sentences: list of tokenized sentences
        vector_size: size of word embeddings
        min_count: ignore words with freq lower than this
        window: max distance between target and context word
        negative: number of negative samples
        cbow: True for CBOW, False for Skip-gram
        epochs: number of training iterations
        seed: random seed for reproducibility
        workers: threads to use (1 for deterministic)

    Returns:
        Trained gensim Word2Vec model
    """
    sg = 0 if cbow else 1

    # model built and trained in one step for deterministic behavior
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        seed=seed,
        workers=workers,
        epochs=epochs
    )

    return model
