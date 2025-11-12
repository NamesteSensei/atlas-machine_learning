#!/usr/bin/env python3
"""
Main file for testing the Word2Vec model training function.
"""

from gensim.test.utils import common_texts
word2vec_model = __import__('2-word2vec').word2vec_model

# /** Display first two example sentences **/
print(common_texts[:2])

# /** Train Word2Vec model with example corpus **/
w2v = word2vec_model(common_texts, min_count=1)

# /** Print vector representation for the word 'computer' **/
print(w2v.wv["computer"])
