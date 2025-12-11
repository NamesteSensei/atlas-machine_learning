#!/usr/bin/env python3
"""
Dataset class for tokenizing the TED HRLR Portuguese–English dataset.

Task 2:
- Add tf_encode() as a TensorFlow wrapper for encode()
- Tokenize the dataset inside __init__
"""

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    Loads and prepares the TED HRLR dataset for machine translation.

    Attributes:
        data_train: Tokenized training dataset.
        data_valid: Tokenized validation dataset.
        tokenizer_pt: Portuguese tokenizer.
        tokenizer_en: English tokenizer.
        start_token (int): Shared start-of-sentence token index.
        end_token (int): Shared end-of-sentence token index.
    """

    def __init__(self):
        """Initialize dataset, tokenizers, and tf.data mappings."""
        raw_train = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="train",
            as_supervised=True
        )

        raw_valid = tfds.load(
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

        self.start_token = self.tokenizer_pt.vocab_size
        self.end_token = self.start_token + 1

        self.data_train = raw_train.map(self.tf_encode)
        self.data_valid = raw_valid.map(self.tf_encode)

    def encode(self, pt, en):
        """
        Tokenize sentences using the Python tokenizers.

        Args:
            pt (tf.Tensor): Portuguese text tensor.
            en (tf.Tensor): English text tensor.

        Returns:
            tuple: Two Python lists of token IDs.
        """
        pt_text = pt.numpy().decode("utf-8")
        en_text = en.numpy().decode("utf-8")

        pt_ids = self.tokenizer_pt.encode(pt_text, add_special_tokens=False)
        en_ids = self.tokenizer_en.encode(en_text, add_special_tokens=False)

        pt_tokens = [self.start_token] + pt_ids + [self.end_token]
        en_tokens = [self.start_token] + en_ids + [self.end_token]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper around encode().

        Args:
            pt (tf.Tensor): Portuguese text.
            en (tf.Tensor): English text.

        Returns:
            tuple: Two tf.Tensor objects of dtype int64.
        """
        pt_out, en_out = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_out.set_shape([None])
        en_out.set_shape([None])

        return pt_out, en_out
