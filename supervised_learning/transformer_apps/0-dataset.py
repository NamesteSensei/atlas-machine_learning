#!/usr/bin/env python3
"""Dataset class for TED HRLR Portuguese-English translation."""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Loads TED HRLR translation dataset and tokenizers."""

    def __init__(self):
        """Initialize training and validation data, and tokenizers."""
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
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates tokenizers for Portuguese and English.

        Args:
            data: tf.data.Dataset, examples as tuples (pt, en)

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )
        return tokenizer_pt, tokenizer_en
