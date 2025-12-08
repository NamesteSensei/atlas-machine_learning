#!/usr/bin/env python3
"""Dataset class with encode wrapper for Transformer translation"""

import tensorflow_datasets as tfds
import tensorflow as tf


class Dataset:
    """
    Loads and prepares the Portuguese-English translation dataset
    for Transformer-based models.
    """

    def __init__(self):
        """
        Initializes the dataset by loading the TED HRLR data and building
        tokenizers. Also maps the dataset to its encoded form using tf_encode.
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

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Builds SubwordTextEncoder tokenizers for Portuguese and English.

        Args:
            data (tf.data.Dataset): Dataset of sentence pairs.

        Returns:
            tuple: (tokenizer_pt, tokenizer_en)
        """
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data),
            target_vocab_size=2 ** 13
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data),
            target_vocab_size=2 ** 13
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes Portuguese and English sentences using subword tokenizers
        and adds start/end tokens.

        Args:
            pt (tf.Tensor): Portuguese sentence.
            en (tf.Tensor): English sentence.

        Returns:
            tuple: Tokenized pt and en tensors.
        """
        pt_tokens = self.tokenizer_pt.encode(pt.numpy())
        en_tokens = self.tokenizer_en.encode(en.numpy())

        pt_tokens = [self.tokenizer_pt.vocab_size] + pt_tokens + [
            self.tokenizer_pt.vocab_size + 1
        ]
        en_tokens = [self.tokenizer_en.vocab_size] + en_tokens + [
            self.tokenizer_en.vocab_size + 1
        ]

        return (
            tf.convert_to_tensor(pt_tokens, dtype=tf.int64),
            tf.convert_to_tensor(en_tokens, dtype=tf.int64)
        )

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper around the encode method using tf.py_function.

        Args:
            pt (tf.Tensor): Portuguese sentence.
            en (tf.Tensor): English sentence.

        Returns:
            tuple: Encoded pt and en tensors with shapes set.
        """
        result_pt, result_en = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
