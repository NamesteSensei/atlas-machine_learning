#!/usr/bin/env python3
"""
Main file used for testing either the bag_of_words or tf_idf functions.
"""

try:
    # Try to import tf_idf (Task 1)
    embedding_func = __import__('1-tf_idf').tf_idf
except Exception:
    # Fallback to bag_of_words (Task 0)
    embedding_func = __import__('0-bag_of_words').bag_of_words

sentences = [
    "Holberton school is Awesome!",
    "Machine learning is awesome",
    "NLP is the future!",
    "The children are our future",
    "Our children's children are our grandchildren",
    "The cake was not very good",
    "No one said that the cake was not very good",
    "Life is beautiful"
]

E, F = embedding_func(sentences)
print(E)
print(F)
