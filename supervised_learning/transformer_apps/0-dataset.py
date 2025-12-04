#!/usr/bin/env python3
"""Dataset class for machine translation using TED Talks pt_to_en"""
import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """
    Loads and prepares dataset for machine translation using pre-trained tokenizers.
    """

    def __init__(self):
        """
        Initializes Dataset instance:
        - Loads train and validation splits
        - Creates subword tokenizers for both Portuguese and English
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates subword tokenizers for the dataset using pre-trained models.
        Args:
            data: tf.data.Dataset, tuples of (pt, en)
        Returns:
            Tuple of (Portuguese tokenizer, English tokenizer)
        """
        tokenizer_pt = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased", use_fast=True)
        tokenizer_en = AutoTokenizer.from_pretrained(
            "bert-base-uncased", use_fast=True)
        return tokenizer_pt, tokenizer_en
