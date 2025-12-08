#!/usr/bin/env python3
"""Dataset class with batching and filtering"""

import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """Loads data and tokenizers"""

    def __init__(self, batch_size, max_len):
        """Initializes dataset, tokenizers, and pipeline"""
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

        self.tokenizer_pt = AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        self.tokenizer_en = AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        self.data_train = self.data_train.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len, tf.size(en) <= max_len
            )
        )
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(
            batch_size, padding_values=(0, 0)
        )
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        self.data_valid = self.data_valid.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len, tf.size(en) <= max_len
            )
        )
        self.data_valid = self.data_valid.padded_batch(
            batch_size, padding_values=(0, 0)
        )

    def encode(self, pt, en):
        """Encodes sentences with start and end tokens"""
        pt_ids = self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8'),
            add_special_tokens=True
        )
        en_ids = self.tokenizer_en.encode(
            en.numpy().decode('utf-8'),
            add_special_tokens=True
        )
        return tf.convert_to_tensor(pt_ids, dtype=tf.int64), \
            tf.convert_to_tensor(en_ids, dtype=tf.int64)

    def tf_encode(self, pt, en):
        """Wrapper for tf.py_function for encoding"""
        result_pt, result_en = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=(tf.int64, tf.int64)
        )
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
