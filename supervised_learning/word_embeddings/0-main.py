#!/usr/bin/env python3
"""Test file for Bag‑of‑Words."""

import importlib.util

# Dynamically import 0‑bag_of_words.py
spec = importlib.util.spec_from_file_location(
    "bow_module", "./0-bag_of_words.py")
bow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bow)
bag_of_words = bow.bag_of_words

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

E, F = bag_of_words(sentences)
print(E)
print(F)

