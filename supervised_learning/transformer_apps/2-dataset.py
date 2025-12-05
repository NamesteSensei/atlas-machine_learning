#!/usr/bin/env python3
"""Dataset with encode wrapper"""

import tensorflow_datasets as tfds
import tensorflow as tf
import transformers


class Dataset:
    """Loads data and tokenizers"""

    def __init__(self):
        """Sets up data pipeline"""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train', as_supervised=True)

        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation', as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Builds subword tokenizers"""
        tok_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data),
            target_vocab_size=2**13)

        tok_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data),
            target_vocab_size=2**13)

        return tok_pt, tok_en

    def encode(self, pt, en):
        """Adds start and end tokens"""
        pt_tokens = self.tokenizer_pt.encode(pt.numpy())
        en_tokens = self.tokenizer_en.encode(en.numpy())

        pt_tokens = [self.tokenizer_pt.vocab_size] + pt_tokens + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + en_tokens + [self.tokenizer_en.vocab_size + 1]

        return tf.convert_to_tensor(pt_tokens, dtype=tf.int64), tf.convert_to_tensor(en_tokens, dtype=tf.int64)

    def tf_encode(self, pt, en):
        """Wrapper for encode"""
        pt_encoded, en_encoded = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])
        return pt_encoded, en_encoded

