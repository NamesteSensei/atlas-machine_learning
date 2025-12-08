#!/usr/bin/env python3
"""Loads and preprocesses the dataset for Transformer training"""

import tensorflow_datasets as tfds
import tensorflow as tf
from transformers import AutoTokenizer


class Dataset:
    """Loads data and tokenizers"""

    def __init__(self, batch_size, max_len):
        """Initialize and preprocess dataset"""
        self.batch_size = batch_size
        self.max_len = max_len

        self.tokenizer_pt = AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        self.tokenizer_en = AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train', as_supervised=True)
        data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation', as_supervised=True)

        data_train = data_train.map(self.tf_encode)
        data_valid = data_valid.map(self.tf_encode)

        data_train = data_train.filter(self.filter_max_len)
        data_train = data_train.cache()
        data_train = data_train.shuffle(20000)
        data_train = data_train.padded_batch(
            self.batch_size, padded_shapes=([None], [None]))
        data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)

        data_valid = data_valid.filter(self.filter_max_len)
        data_valid = data_valid.padded_batch(
            self.batch_size, padded_shapes=([None], [None]))

        self.data_train = data_train
        self.data_valid = data_valid

    def encode(self, pt, en):
        """Encodes sentences with start and end tokens"""
        pt_tokens = self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8'),
            add_special_tokens=True)
        en_tokens = self.tokenizer_en.encode(
            en.numpy().decode('utf-8'),
            add_special_tokens=True)
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TF wrapper for encode"""
        result_pt, result_en = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en

    def filter_max_len(self, pt, en):
        """Filter out sentence pairs longer than max_len"""
        return tf.logical_and(
            tf.size(pt) <= self.max_len,
            tf.size(en) <= self.max_len)
