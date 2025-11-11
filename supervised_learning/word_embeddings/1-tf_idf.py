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
    # Tokenize and clean sentences
    tokenized = [preprocess(s) for s in sentences]

    # Build vocabulary
    if vocab is None:
        vocab_set = set()
        for words in tokenized:
            vocab_set.update(words)
        vocab = sorted(vocab_set)

    features = np.array(vocab)
    s = len(sentences)
    f = len(vocab)

    # Initialize matrices
    tf = np.zeros((s, f))
    idf = np.zeros(f)

    # Compute Term Frequency (TF)
    for i, words in enumerate(tokenized):
        for word in words:
            if word in vocab:
                j = vocab.index(word)
                tf[i][j] += 1
        if len(words) > 0:
            tf[i] = tf[i] / len(words)

    # Compute Inverse Document Frequency (IDF)
    for j, word in enumerate(vocab):
        doc_count = sum(word in words for words in tokenized)
        if doc_count > 0:
            idf[j] = np.log(s / doc_count)
        else:
            idf[j] = 0.0

    # Compute TF-IDF
    embeddings = tf * idf

    # Normalize embeddings (L2 norm)
    for i in range(s):
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] /= norm

    return embeddings, features
