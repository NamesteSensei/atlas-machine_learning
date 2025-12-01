#!/usr/bin/env python3
"""
Dataset module for loading and tokenizing the TED HRLR Portuguese-to-English
translation dataset using TensorFlow Datasets and HuggingFace tokenizers.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """
    Loads and prepares the dataset for machine translation.
    """

    def __init__(self):
        """
        Class constructor.

        Creates the instance attributes:
            - data_train: training split
            - data_valid: validation split
            - tokenizer_pt: Portuguese tokenizer
            - tokenizer_en: English tokenizer
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

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates subword tokenizers for the dataset.

        Args:
            data (tf.data.Dataset): dataset containing (pt, en) sentence pairs

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        tokenizer_pt = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )
        return tokenizer_pt, tokenizer_en
