#!/usr/bin/env python3
"""
Dataset class for the TED HRLR Portuguese-to-English translation dataset.
Loads the training and validation splits and initializes the tokenizers.
"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Loads and prepares the TED HRLR translation dataset.

    Attributes:
        data_train: Training split of the dataset.
        data_valid: Validation split of the dataset.
        tokenizer_pt: Portuguese tokenizer.
        tokenizer_en: English tokenizer.
    """

    def __init__(self):
        """
        Initializes dataset splits and tokenizers.
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

        (self.tokenizer_pt,
         self.tokenizer_en) = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates the Portuguese and English tokenizers.

        Args:
            data: Dataset containing paired text samples.

        Returns:
            tokenizer_pt: Portuguese tokenizer.
            tokenizer_en: English tokenizer.
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )

        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        return tokenizer_pt, tokenizer_en
