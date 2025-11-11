#!/usr/bin/env python3
"""Creates a Bag‑of‑Words (BoW) embedding matrix."""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Builds a BoW matrix from a list of sentences.
    Args:
        sentences (list): sentences to analyze
        vocab (list): optional vocabulary; if None, built from sentences
    Returns:
        embeddings (ndarray): shape (s, f) word‑count matrix
        features (list): list of vocabulary terms
    """
    def clean_text(text):
        """Lowercase and remove punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    # Tokenize and clean
    tokenized = [clean_text(s) for s in sentences]

    # Build vocabulary if none provided
    if vocab is None:
        vocab = sorted({w for sent in tokenized for w in sent})

    # Word index map
    w_idx = {w: i for i, w in enumerate(vocab)}

    # Initialize zero matrix
    mat = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Count word occurrences
    for i, sent in enumerate(tokenized):
        for w in sent:
            if w in w_idx:
                mat[i, w_idx[w]] += 1

    return mat, vocab
