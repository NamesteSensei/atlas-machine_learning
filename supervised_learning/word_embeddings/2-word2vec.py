#!/usr/bin/env python3
"""
Train a deterministic Word2Vec model.
"""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    Creates and trains a Word2Vec model.

    Args:
        sentences (list of list of str): tokenized sentences.
        vector_size (int): embedding dimension.
        min_count (int): minimum word frequency.
        window (int): context window size.
        negative (int): negative samples.
        cbow (bool): True = CBOW, False = Skip-gram.
        epochs (int): number of epochs.
        seed (int): random seed.
        workers (int): number of threads.

    Returns:
        gensim.models.Word2Vec: trained model.
    """
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
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

    # Match gensim 4.1.2 vocabulary order
    model.wv.sort_by_descending_frequency()

    return model
