#!/usr/bin/env python3
"""
Dataset loader and tokenizer for Portuguese to English translation using
TensorFlow Datasets and Hugging Face Transformers.
"""

import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """
    Loads and preprocesses the TED Talks pt_to_en dataset for translation.
    """

    def __init__(self):
        """
        Initializes the Dataset instance by loading the train and validation
        splits, and preparing the tokenizers.
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
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Loads pre-trained BERT tokenizers for both Portuguese and English.

        Args:
            data (tf.data.Dataset): Dataset containing (pt, en) sentence pairs.

        Returns:
            tuple: (tokenizer_pt, tokenizer_en)
        """
        tokenizer_pt = AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            use_fast=True
        )
        tokenizer_en = AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            use_fast=True
        )
        return tokenizer_pt, tokenizer_en
