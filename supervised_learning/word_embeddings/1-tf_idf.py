#!/usr/bin/env python3
"""
This module defines the tf_idf function that generates a TF-IDF
embedding matrix for a list of sentences using a provided vocabulary
or the full word set.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding for a list of sentences.

    Args:
        sentences (list): List of sentences to analyze.
        vocab (list): List of vocabulary words to use for analysis.
                      If None, all words within the sentences are used.

    Returns:
        embeddings (np.ndarray): Array of shape (s, f) containing embeddings.
                                 s = number of sentences
                                 f = number of features analyzed.
        features (list): List of feature words used for embeddings.
    """
    # /** Create the TF-IDF vectorizer object **/
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # /** Fit the vectorizer on the input sentences and transform them **/
    X = vectorizer.fit_transform(sentences)

    # /** Extract the feature names (vocabulary terms actually used) **/
    features = vectorizer.get_feature_names_out()

    # /** Convert the sparse matrix into a dense NumPy array **/
    embeddings = X.toarray()

    # /** Return the dense embeddings matrix and the feature list **/
    return embeddings, features
