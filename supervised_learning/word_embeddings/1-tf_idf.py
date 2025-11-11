#!/usr/bin/env python3
"""
Module that creates a TF-IDF embedding matrix from a list of sentences.
"""

import re
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel


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
        List of lowercase words with punctuation removed.
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
        Vocabulary words to use for analysis.
        If None, all words within the sentences are used.

    Returns
    -------
    embeddings : numpy.ndarray
        Array of shape (s, f) containing the embeddings,
        where s is the number of sentences and f is the number of features.
    features : numpy.ndarray
        Array of vocabulary words used for embeddings.
    """
    # Tokenize and clean sentences
    tokenized = [preprocess(s) for s in sentences]

    # Create dictionary mapping words to IDs
    dictionary = Dictionary(tokenized)

    # Restrict dictionary to provided vocabulary, if any
    if vocab is not None:
        # Determine tokens not in vocab and remove them
        remove_ids = [dictionary.token2id[w]
                      for w in list(dictionary.token2id.keys())
                      if w not in vocab]
        dictionary.filter_tokens(remove_ids)
        dictionary.compactify()

    # Convert tokenized sentences to bag-of-words format
    corpus = [dictionary.doc2bow(text) for text in tokenized]

    # Build TF-IDF model
    tfidf_model = TfidfModel(corpus)

    # Compute TF-IDF values for each document
    tfidf_docs = [tfidf_model[doc] for doc in corpus]

    # Determine feature list (vocabulary)
    vocab_list = vocab if vocab is not None else list(dictionary.token2id.keys())
    features = np.array(vocab_list)
    embeddings = np.zeros((len(sentences), len(features)))

    # Populate embeddings matrix
    for i, doc in enumerate(tfidf_docs):
        for word_id, value in doc:
            word = dictionary[word_id]
            if word in vocab_list:
                j = vocab_list.index(word)
                embeddings[i, j] = value

    return embeddings, features
