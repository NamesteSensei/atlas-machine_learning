#!/usr/bin/env python3
"""Dataset class for Portuguese to English translation"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Loads and prepares the translation dataset"""

    def __init__(self):
        """Loads training and validation data, sets tokenizers"""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train', as_supervised=True)
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Creates tokenizers from pre-trained models"""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased", use_fast=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased", use_fast=True)
        return tokenizer_pt, tokenizer_en
