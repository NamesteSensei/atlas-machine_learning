#!/usr/bin/env python3
"""
Module for handling a bilingual text dataset.
This class loads the training and validation pairs
and prepares tokenizers for both languages.
"""
import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """
    Prepares the dataset needed for language translation.
    """

    def __init__(self):
        """
        Initializes the training and validation data.
        Also builds the tokenizers.
        """
        self.data_train = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="train",
            as_supervised=True
        )

        self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="validation",
            as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Builds tokenizers for both languages.

        Args:
            data: dataset of paired sentences

        Returns:
            tokenizer_pt: tokenizer for the first language
            tokenizer_en: tokenizer for the second language
        """
        tokenizer_pt = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )

        tokenizer_en = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        return tokenizer_pt, tokenizer_en
