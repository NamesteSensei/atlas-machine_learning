#!/usr/bin/env python3
"""
Train a FastText embedding model.
"""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5,
                   negative=5, window=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    Creates and trains a FastText model.

    Args:
        sentences (list of list of str): tokenized sentences.
        vector_size (int): embedding dimension.
        min_count (int): minimum word frequency.
        negative (int): negative sampling.
        window (int): context window size.
        cbow (bool): True = CBOW, False = Skip-gram.
        epochs (int): number of epochs.
        seed (int): random seed.
        workers (int): number of threads.

    Returns:
        gensim.models.FastText: trained FastText model.
    """
    sg = 0 if cbow else 1

    model = gensim.models.FastText(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
