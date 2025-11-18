#!/usr/bin/env python3
"""Test Task 1: N-gram BLEU score"""

ngram_bleu = __import__('1-ngram_bleu').ngram_bleu

references = [
    ["the", "cat", "is", "on", "the", "mat"],
    ["there", "is", "a", "cat", "on", "the", "mat"]
]

sentence = ["there", "is", "a", "cat", "here"]

print(ngram_bleu(references, sentence, 2))
