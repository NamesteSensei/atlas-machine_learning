#!/usr/bin/env python3
"""
Module that creates, builds, and trains a Word2Vec model.
"""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    Creates and trains a gensim Word2Vec model.

    Args:
        sentences (list of list of str): tokenized sentences to train on.
        vector_size (int): dimensionality of the word vectors.
        min_count (int): minimum number of occurrences for a word to be
            considered in training.
        window (int): maximum distance between the current and predicted
            word within a sentence.
        negative (int): number of negative samples for negative sampling.
        cbow (bool): True to use CBOW; False to use skip-gram.
        epochs (int): number of training epochs.
        seed (int): random seed for reproducibility.
        workers (int): number of worker threads to train the model.

    Returns:
        gensim.models.Word2Vec: the trained Word2Vec model.
    """
    # sg = 0 → CBOW, sg = 1 → skip-gram
    sg = 0 if cbow else 1

    # Create the model without training yet
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    # Build vocabulary from the provided sentences
    model.build_vocab(sentences)

    # Train the model for the specified number of epochs
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
