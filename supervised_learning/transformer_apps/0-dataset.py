#!/usr/bin/env python3
"""
Handles dataset loading and tokenization for bilingual translation.
Uses tfds to load data and transformers to create tokenizers.
"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Prepares training/validation splits and tokenizer objects.
    """

    def __init__(self):
        """
        Loads dataset splits and builds tokenizers.
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
        Builds tokenizers using pretrained BERT models.

        Args:
            data: dataset containing pairs of text samples

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
