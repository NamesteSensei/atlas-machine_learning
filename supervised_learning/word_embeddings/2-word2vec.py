#!/usr/bin/env python3
"""Word2Vec Model Trainer Module"""

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
    /**
     * Trains a Word2Vec model using Gensim.
     *
     * PARAMETERS:
     * sentences  → list of tokenized sentences to train on.
     * vector_size → dimensionality of embedding vectors.
     * min_count   → ignore words with frequency lower than this value.
     * window      → max distance between current and predicted word.
     * negative    → number of negative samples for training.
     * cbow        → True for CBOW; False for Skip-gram.
     * epochs      → number of iterations to train over the corpus.
     * seed        → random seed for reproducibility.
     * workers     → number of worker threads to train the model.
     *
     * RETURNS:
     * Trained gensim Word2Vec model.
     */
    """

    """Set training algorithm type: 0 for CBOW, 1 for Skip-gram"""
    sg = 0 if cbow else 1

    """
    Create and configure a Word2Vec model.
    """
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        seed=seed
    )

    """Build vocabulary from the provided list of tokenized sentences"""
    model.build_vocab(sentences)

    """Train model using corpus with defined number of epochs"""
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    """Return the trained Word2Vec model"""
    return model
