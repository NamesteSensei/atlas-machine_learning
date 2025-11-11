#!/usr/bin/env python3
"""
Module that creates a bag of words embedding matrix from a list of sentences.
"""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Parameters
    ----------
    sentences : list
        List of sentences to analyze.
    vocab : list, optional
        List of the vocabulary words to use for the analysis.
        If None, all words within the sentences will be used.

    Returns
    -------
    embeddings : numpy.ndarray
        Array of shape (s, f) containing the embeddings,
        where s is the number of sentences and f is the number of features.
    features : numpy.ndarray
        Array of the features (vocabulary words) used for the embeddings.
    """
    tokenized = []
    for sentence in sentences:
        cleaned = re.sub(r"\'s\b", "", sentence.lower())
        cleaned = re.sub(r"[^a-z\s]", "", cleaned)
        words = cleaned.split()
        tokenized.append(words)

    if vocab is None:
        vocab_set = set()
        for words in tokenized:
            vocab_set.update(words)
        vocab = sorted(vocab_set)

    word_index = {word: idx for idx, word in enumerate(vocab)}
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, words in enumerate(tokenized):
        for word in words:
            if word in word_index:
                embeddings[i][word_index[word]] += 1

    return embeddings, np.array(vocab)
