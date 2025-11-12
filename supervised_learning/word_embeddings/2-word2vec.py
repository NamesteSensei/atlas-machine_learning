#!/usr/bin/env python3
"""
This module defines the word2vec_model function that builds
and trains a Word2Vec model using gensim.
"""

from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim Word2Vec model.

    Args:
        sentences (list): List of sentences to be trained on.
        vector_size (int): Dimensionality of the embedding layer.
        min_count (int): Minimum number of occurrences of a word
                         for use in training.
        window (int): Maximum distance between the current and
                      predicted word within a sentence.
        negative (int): Size of negative sampling.
        cbow (bool): True trains CBOW; False trains Skip-gram.
        epochs (int): Number of iterations to train over.
        seed (int): Random seed for reproducibility.
        workers (int): Number of worker threads to train the model.

    Returns:
        model (Word2Vec): The trained Word2Vec model.
    """
    # /** Select training algorithm: CBOW (sg=0) or Skip-gram (sg=1) **/
    sg = 0 if cbow else 1

    # /** Initialize Word2Vec model with provided parameters **/
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    # /** Train the model for the given number of epochs **/
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    # /** Return the trained Word2Vec model **/
    return model
