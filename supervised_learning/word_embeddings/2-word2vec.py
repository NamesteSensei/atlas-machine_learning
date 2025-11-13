#!/usr/bin/env python3
"""Word2Vec model trainer using gensim"""

from gensim.models import Word2Vec

def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """
    Trains a Word2Vec model on the given sentences
    using gensim.

    Parameters:
    - sentences: list of tokenized sentences
    - vector_size: output vector size
    - min_count: minimum frequency of words
    - window: max distance between current and predicted word
    - negative: negative sampling size
    - cbow: True for CBOW; False for Skip-gram
    - epochs: number of training epochs
    - seed: random seed
    - workers: number of threads

    Returns:
    - trained Word2Vec model
    """

    sg = 0 if cbow else 1  # Skip-gram if cbow=False

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        seed=seed,
        workers=workers
    )

    model.train(sentences, total_examples=len(sentences), epochs=epochs)

    return model
