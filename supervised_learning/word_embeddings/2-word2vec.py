#!/usr/bin/env python3
"""Word2Vec Model Trainer"""

from gensim.models import Word2Vec

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
     * Function to create, build, and train a Gensim Word2Vec model.
     *
     * PARAMETERS:
     * sentences  → list of tokenized sentences to train on.
     * vector_size → dimension of the word embedding vectors.
     * min_count   → ignores words with total frequency lower than this.
     * window      → maximum distance between target word and its context.
     * negative    → number of negative samples (for negative sampling).
     * cbow        → training algorithm: True for CBOW, False for Skip-gram.
     * epochs      → number of training iterations.
     * seed        → random seed for reproducibility.
     * workers     → number of CPU threads to train with.
     *
     * RETURNS:
     * The trained Word2Vec model.
     */
    """

    """/** Set training algorithm type: 0 for CBOW, 1 for Skip-gram **/"""
    sg = 0 if cbow else 1

    """
    /** 
     * Initialize the Word2Vec model with parameters.
     * We do NOT train yet; we just build the model configuration.
     */
    """
    model = Word2Vec(
        vector_size=vector_size,   # number of features in the word vectors
        window=window,             # size of context window
        min_count=min_count,       # ignore rare words
        workers=workers,           # CPU threads
        sg=sg,                     # 0=CBOW, 1=Skip-gram
        negative=negative,         # negative sampling
        seed=seed                  # random seed
    )

    """/** Build vocabulary from training sentences **/"""
    model.build_vocab(sentences)

    """/** Train the model using the prepared vocabulary **/"""
    model.train(
        sentences, 
        total_examples=model.corpus_count,
        epochs=epochs
    )

    """/** Return the trained Word2Vec model **/"""
    return model
