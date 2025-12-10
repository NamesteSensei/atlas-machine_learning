#!/usr/bin/env python3
"""
Dataset class for loading TED HRLR Portuguese–English translation data.

This class loads the train and validation splits and initializes two BERT
tokenizers (Portuguese and English).
"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Loads and prepares the TED HRLR dataset for machine translation.

    Attributes:
        data_train (tf.data.Dataset): Training dataset split.
        data_valid (tf.data.Dataset): Validation dataset split.
        tokenizer_pt: Portuguese BERT tokenizer.
        tokenizer_en: English BERT tokenizer.
    """

    def __init__(self):
        """Initialize dataset splits and tokenizers."""
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

        # BERT multilingual tokenizer (best choice for pt & en)
        self.tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )

        self.tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )
