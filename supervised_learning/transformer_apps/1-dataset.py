#!/usr/bin/env python3
"""
Dataset class for loading and encoding TED HRLR Portuguese–English data.

This version adds the encode() method required for task 1.
"""

import tensorflow_datasets as tfds
import transformers
import numpy as np
import tensorflow as tf


class Dataset:
    """
    Loads and prepares the TED HRLR dataset for machine translation.

    Attributes:
        data_train: Training dataset split.
        data_valid: Validation dataset split.
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

        self.tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )
        self.tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )

        # vocab start + end tokens
        self.start_token = self.tokenizer_pt.vocab_size
        self.end_token = self.tokenizer_pt.vocab_size + 1

    def encode(self, pt, en):
        """
        Encode Portuguese and English sentences into token ids.

        Args:
            pt (tf.Tensor): Portuguese sentence.
            en (tf.Tensor): English sentence.

        Returns:
            tuple: (pt_tokens, en_tokens)
                pt_tokens (np.ndarray): Token IDs for Portuguese.
                en_tokens (np.ndarray): Token IDs for English.
        """
        # Convert tensors to python strings
        pt_str = pt.numpy().decode("utf-8")
        en_str = en.numpy().decode("utf-8")

        # Tokenize sentences
        pt_encoded = self.tokenizer_pt.encode(pt_str, add_special_tokens=False)
        en_encoded = self.tokenizer_en.encode(en_str, add_special_tokens=False)

        # Add start and end tokens
        pt_tokens = [self.start_token] + pt_encoded + [self.end_token]
        en_tokens = [self.start_token] + en_encoded + [self.end_token]

        return np.array(pt_tokens), np.array(en_tokens)
