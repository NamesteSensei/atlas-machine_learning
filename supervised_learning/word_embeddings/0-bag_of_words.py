#!/usr/bin/env python3
"""
Creates a bag of words embedding matrix from a list of sentences.
"""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Converts sentences into a bag of words matrix.

    Parameters:
    - sentences: list of strings, each a sentence
    - vocab: optional list of words to use as vocabulary

    Returns:
    - embeddings: numpy.ndarray of shape (s, f)
      where s = number of sentences, f = number of features (words)
    - features: list of vocabulary words used (sorted)
    """
    # /** Clean and tokenize all sentences **/
    tokenized = []
    for sentence in sentences:
        # /** Normalize: lowercase and remove punctuation **/
        words = re.findall(r'\b\w+\b', sentence.lower())
        tokenized.append(words)

    # /** Build vocabulary if not provided **/
    if vocab is None:
        vocab_set = set()
        for words in tokenized:
            vocab_set.update(words)
        vocab = sorted(vocab_set)  # /** Sort for consistent feature order **/

    # /** Create word index for feature lookup **/
    word_index = {word: idx for idx, word in enumerate(vocab)}

    # /** Initialize embedding matrix **/
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    # /** Populate embedding matrix with word counts **/
    for i, words in enumerate(tokenized):
        for word in words:
            if word in word_index:
                embeddings[i][word_index[word]] += 1

    return embeddings, vocab
