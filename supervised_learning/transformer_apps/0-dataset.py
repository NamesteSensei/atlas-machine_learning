#!/usr/bin/env python3
"""
Loads training text and prepares token tools.
"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Manages two language inputs and token tools.
    """

    def __init__(self):
        """
        Sets up training and validation pairs and token tools.
        """
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self._load_tokens()

    def _load_tokens(self):
        """
        Creates tools that map text to numbers.

        Returns:
            tuple: tools for Portuguese and English
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            use_fast=True
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            use_fast=True
        )
        return tokenizer_pt, tokenizer_en
