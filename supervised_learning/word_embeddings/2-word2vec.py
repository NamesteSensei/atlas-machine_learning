#!/usr/bin/env python3
"""
Creates, builds, and trains a gensim Word2Vec model.

Constraints:
- Only 'import gensim' is used (checker requirement).
- Function is deterministic as long as environment (gensim/numpy/BLAS) is fixed.
"""

import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True,
                   epochs=5, seed=0, workers=1):
    """
    Create and train a gensim Word2Vec model.

    Args:
        sentences (list[list[str]]): tokenized sentences.
        vector_size (int): embedding dimensionality.
        min_count (int): min word frequency to keep.
        window (int): context window size.
        negative (int): negative sampling size.
        cbow (bool): True => CBOW, False => skip-gram.
        epochs (int): training epochs.
        seed (int): RNG seed (important for reproducibility).
        workers (int): number of worker threads (use 1 for determinism).

    Returns:
        gensim.models.Word2Vec: trained model.
    """
    # sg: 0 for CBOW, 1 for skip-gram (gensim API)
    sg = 0 if cbow else 1

    # Create an uninitialized model object (no implicit vocab/train).
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        workers=workers
    )

    # Explicitly build the vocabulary from sentences
    model.build_vocab(sentences)

    # Train the model with explicit total_examples and epochs
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
