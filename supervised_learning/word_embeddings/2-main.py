#!/usr/bin/env python3

from gensim.test.utils import common_texts
word2vec_model = __import__('2-word2vec').word2vec_model

print(common_texts[:2])  # Show first 2 sample sentences

w2v = word2vec_model(common_texts, min_count=1)  # Train model
print(w2v.wv["computer"])  # Print word vector for 'computer'
