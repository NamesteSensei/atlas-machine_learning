#!/usr/bin/env python3
"""
Module that creates a TF-IDF embedding matrix from a list of sentences.
"""

import numpy as np
import re


def preprocess(sentence):
    """
    Cleans and tokenizes a sentence into lowercase words.

    Parameters
    ----------
    sentence : str
        Input sentence to process.

    Returns
    -------
    list
        List of lowercase words without punctuation or possessives.
    """
    sentence = re.sub(r"\'s\b", "", sentence.lower())
    sentence = re.sub(r"[^a-z\s]", "", sentence)
    return sentence.split()


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Parameters
    ----------
    sentences : list
        List of sentences to analyze.
    vocab : list, optional
        List of vocabulary words to use for the analysis.
        If None, all words within the sentences are used.

    Returns
    -------
    embeddings : numpy.ndarray
        Array of shape (s, f) containing the embeddings.
    features : numpy.ndarray
        Array of the features (vocabulary words) used for embeddings.
    """
    # Tokenize sentences
    tokenized = [preprocess(s) for s in sentences]

    # Build or use given vocabulary
    if vocab is None:
        vocab_set = set()
        for words in tokenized:
            vocab_set.update(words)
        vocab = sorted(vocab_set)

    features = np.array(vocab)
    s = len(sentences)
    f = len(vocab)

    # Term Frequency (TF)
    tf = np.zeros((s, f))
    for i, words in enumerate(tokenized):
        total = len(words)
        for word in words:
            if word in vocab:
                j = vocab.index(word)
                tf[i][j] += 1
        if total > 0:
            tf[i] /= total

    # Inverse Document Frequency (IDF)
    idf = np.zeros(f)
    for j, word in enumerate(vocab):
        doc_count = sum(word in words for words in tokenized)
        if doc_count > 0:
            idf[j] = np.log10(s / doc_count) + 1  # log base10 + 1
        else:
            idf[j] = 0

    # TF-IDF matrix
    embeddings = tf * idf

    # L2-normalize each row
    for i in range(s):
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] /= norm

    return embeddings, features
