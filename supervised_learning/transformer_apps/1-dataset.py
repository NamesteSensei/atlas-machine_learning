#!/usr/bin/env python3
"""Dataset with encode"""

import tensorflow_datasets as tfds
import transformers
import numpy as np


class Dataset:
    """Loads data and tokenizers"""

    def __init__(self):
        """Loads train and validation data"""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train', as_supervised=True)

        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation', as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Creates pretrained tokenizers"""
        tok_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            use_fast=True)

        tok_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            use_fast=True)

        return tok_pt, tok_en

    def encode(self, pt, en):
        """Encodes sentences with start and end tokens"""
        pt_tokens = self.tokenizer_pt.encode(
            pt.numpy().decode("utf-8"),
            add_special_tokens=False)

        en_tokens = self.tokenizer_en.encode(
            en.numpy().decode("utf-8"),
            add_special_tokens=False)

        pt_start = self.tokenizer_pt.vocab_size
        pt_end = pt_start + 1

        en_start = self.tokenizer_en.vocab_size
        en_end = en_start + 1

        pt_out = [pt_start] + pt_tokens + [pt_end]
        en_out = [en_start] + en_tokens + [en_end]

        return np.array(pt_out), np.array(en_out)
