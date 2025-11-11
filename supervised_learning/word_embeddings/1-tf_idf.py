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

    # Build vocabulary if not provided
    if vocab is None:
        vocab = sorted({word for sent in tokenized for word in sent})

    features = np.array(vocab)
    s = len(sentences)
    f = len(vocab)

    # Initialize TF and IDF arrays
    tf = np.zeros((s, f))
    idf = np.zeros(f)

    # --- Term Frequency (TF) ---
    for i, words in enumerate(tokenized):
        for word in words:
            if word in vocab:
                tf[i][vocab.index(word)] += 1
        total = len(words)
        if total > 0:
            tf[i] /= total

    # --- Inverse Document Frequency (IDF) ---
    for j, word in enumerate(vocab):
        doc_count = sum(word in sent for sent in tokenized)
        if doc_count > 0:
            idf[j] = 1 + np.log(s / doc_count)  # natural log
        else:
            idf[j] = 0

    # --- Compute TF-IDF ---
    embeddings = tf * idf

    # --- L2 Normalization ---
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings, features
