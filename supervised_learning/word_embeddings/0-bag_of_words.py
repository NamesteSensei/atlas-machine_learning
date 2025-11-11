#!/usr/bin/env python3
"""Creates a Bag of Words embedding matrix."""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a Bag of Words (BoW) embedding matrix.
    Args:
        sentences (list): list of sentences to analyze
        vocab (list): list of vocabulary words to use
    Returns:
        embeddings (ndarray): shape (s, f) BoW matrix
        features (list): list of features (words)
    """
    def clean(text):
        """Lowercase and remove non-alphabetic characters."""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text.split()

    # Clean and tokenize each sentence
    tokenized = [clean(s) for s in sentences]

    # Build vocabulary if not provided
    if vocab is None:
        vocab = sorted({w for sent in tokenized for w in sent})

    # Create mapping word → index
    word_idx = {w: i for i, w in enumerate(vocab)}

    # Initialize zero matrix
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Fill matrix with word counts
    for i, sent in enumerate(tokenized):
        for w in sent:
            if w in word_idx:
                embeddings[i, word_idx[w]] += 1

    return embeddings, vocab
