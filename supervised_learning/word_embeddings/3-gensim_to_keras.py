#!/usr/bin/env python3
"""Module that trains a gensim Word2Vec model."""

import gensim  # /**/ project constraint: only this import is allowed


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """Create, build and train a gensim Word2Vec model.

    Args:
        sentences (list of list of str): Sentences to be trained on, each
            sentence is a list of tokens.
        vector_size (int): Dimensionality of the embedding vectors.
        min_count (int): Minimum number of occurrences for a word to be
            included in the vocabulary.
        window (int): Maximum distance between the current and predicted
            word within a sentence.
        negative (int): Number of negative samples used in negative
            sampling.
        cbow (bool): If True, use CBOW; if False, use Skip-gram.
        epochs (int): Number of passes (iterations) over the training data.
        seed (int): Seed for the random number generator to make training
            reproducible when workers == 1.
        workers (int): Number of worker threads to train the model.

    Returns:
        gensim.models.word2vec.Word2Vec: The trained Word2Vec model.
    """
    # /**/ choose training algorithm: 0 = CBOW, 1 = Skip-gram
    sg = 0 if cbow else 1

    # /**/ create an untrained Word2Vec model with the given hyperparameters
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        seed=seed,
        workers=workers
    )

    # /**/ build the vocabulary from the provided sentences
    model.build_vocab(sentences)

    # /**/ train the model for the requested number of epochs
    model.train(
        sentences,
        total_examples=len(sentences),
        epochs=epochs
    )

    # /**/ return the trained gensim Word2Vec model
    return model
