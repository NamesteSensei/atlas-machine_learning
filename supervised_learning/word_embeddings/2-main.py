#!/usr/bin/env python3
"""Main file for testing the Word2Vec model"""

from gensim.test.utils import common_texts
word2vec_model = __import__('2-word2vec').word2vec_model

print("Sample Sentences:")
print(common_texts[:2])

print("\nModel with seed=1")
w2v_seed1 = word2vec_model(common_texts, min_count=1, seed=1)
print(w2v_seed1.wv["computer"])

print("\nModel with seed=2 and different word='human'")
w2v_seed2 = word2vec_model(common_texts, min_count=1, seed=2)
print(w2v_seed2.wv["human"])
