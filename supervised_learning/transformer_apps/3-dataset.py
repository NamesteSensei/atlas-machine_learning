#!/usr/bin/env python3
"""
Task 3 dataset pipeline for TED HRLR Portuguese–English translation.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Builds dataset pipeline and exposes vocab sizes."""

    def __init__(self, batch_size, max_len):
        train_raw = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="train",
            as_supervised=True,
        )

        self.tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )
        self.tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )

        self.start_token = self.tokenizer_pt.vocab_size
        self.end_token = self.start_token + 1

        self.input_vocab_size = self.tokenizer_pt.vocab_size + 2
        self.target_vocab_size = self.tokenizer_en.vocab_size + 2

        train = train_raw.map(self.tf_encode)
        train = train.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len,
            )
        )
        train = train.shuffle(20000)
        train = train.padded_batch(
            batch_size,
            padded_shapes=([None], [None]),
            padding_values=(
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int64),
            ),
        )
        train = train.prefetch(tf.data.AUTOTUNE)

        self.data_train = train

    def encode(self, pt, en):
        """Tokenize and add start/end tokens."""
        pt_ids = self.tokenizer_pt.encode(
            pt.numpy().decode("utf-8"),
            add_special_tokens=False,
        )
        en_ids = self.tokenizer_en.encode(
            en.numpy().decode("utf-8"),
            add_special_tokens=False,
        )

        return (
            [self.start_token] + pt_ids + [self.end_token],
            [self.start_token] + en_ids + [self.end_token],
        )

    def tf_encode(self, pt, en):
        """TensorFlow wrapper for encode()."""
        pt_out, en_out = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64],
        )
        pt_out.set_shape([None])
        en_out.set_shape([None])
        return pt_out, en_out
