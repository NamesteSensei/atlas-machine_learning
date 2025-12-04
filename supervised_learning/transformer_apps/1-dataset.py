#!/usr/bin/env python3
"""Dataset class for translating Portuguese to English"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow_datasets.deprecated.text import SubwordTextEncoder


class Dataset:
    """Loads and prepares a dataset for machine translation"""

    def __init__(self):
        """Class constructor"""
        examples, _ = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            with_info=True,
            as_supervised=True
        )

        self.data_train = examples['train']
        self.data_valid = examples['validation']

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Creates subword tokenizers for Portuguese and English"""

        pt_sentences = []
        en_sentences = []

        for pt, en in tfds.as_numpy(data):
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            pt_sentences, target_vocab_size=2**13)
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            en_sentences, target_vocab_size=2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encodes Portuguese and English sentences into token IDs"""

        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        pt_tokens = [vocab_size_pt] + self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8')) + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + self.tokenizer_en.encode(
            en.numpy().decode('utf-8')) + [vocab_size_en + 1]

        return np.array(pt_tokens), np.array(en_tokens)
