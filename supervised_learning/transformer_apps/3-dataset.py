#!/usr/bin/env python3
"""
Dataset pipeline for TED HRLR pt-en translation.
Implements:
- encode()
- tf_encode()
- filtering
- batching
- caching
- shuffling
- prefetching
"""

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """Dataset loader, tokenizer setup, and tf.data pipeline."""

    def __init__(self, batch_size, max_len):
        """
        Initialize dataset, tokenizers, and pipeline.

        Args:
            batch_size: training/validation batch size
            max_len: maximum number of tokens allowed per sentence
        """
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

        train = raw_train.map(self.tf_encode)
        valid = raw_valid.map(self.tf_encode)

        train = train.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )
        )

        valid = valid.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )
        )

        train = train.cache()
        train = train.shuffle(20000)

        train = train.padded_batch(
            batch_size,
            padded_shapes=( [None], [None] ),
            padding_values=(tf.cast(0, tf.int64),
                            tf.cast(0, tf.int64))
        )

        valid = valid.padded_batch(
            batch_size,
            padded_shapes=( [None], [None] ),
            padding_values=(tf.cast(0, tf.int64),
                            tf.cast(0, tf.int64))
        )

        train = train.prefetch(tf.data.experimental.AUTOTUNE)

        self.data_train = train
        self.data_valid = valid

    def encode(self, pt, en):
        """Python tokenization function used by tf.py_function."""
        pt_s = pt.numpy().decode("utf-8")
        en_s = en.numpy().decode("utf-8")

        pt_ids = self.tokenizer_pt.encode(pt_s, add_special_tokens=False)
        en_ids = self.tokenizer_en.encode(en_s, add_special_tokens=False)

        pt_out = [self.start_token] + pt_ids + [self.end_token]
        en_out = [self.start_token] + en_ids + [self.end_token]

        return pt_out, en_out

    def tf_encode(self, pt, en):
        """TensorFlow wrapper for encode()."""
        pt_out, en_out = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_out.set_shape([None])
        en_out.set_shape([None])

        return pt_out, en_out
