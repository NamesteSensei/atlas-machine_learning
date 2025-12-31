#!/usr/bin/env python3
"""
Task 3 dataset pipeline for TED HRLR Portuguese–English translation.
Builds filtering, batching, caching, and prefetch.
Allowed only.
"""

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    Builds a full TensorFlow input pipeline.
    Attributes:
        data_train: Training dataset.
        data_valid: Validation dataset.
        tokenizer_pt: Portuguese tokenizer.
        tokenizer_en: English tokenizer.
        start_token: Index for start of sentence.
        end_token: Index for end of sentence.
    """

    def __init__(self, batch_size, max_len):
        """
        Initialize dataset, tokenizers, and pipeline.
        Args:
            batch_size: Batch size.
            max_len: Maximum allowed token count.
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
        train = train.filter(
            lambda pt, en: tf.size(pt) <= max_len
            and tf.size(en) <= max_len
        )
        train = train.cache()
        train = train.shuffle(20000)

        train = train.padded_batch(
            batch_size,
            padded_shapes=([None], [None]),
            padding_values=(
                tf.constant(0, tf.int64),
                tf.constant(0, tf.int64)
            )
        )

        train = train.prefetch(tf.data.experimental.AUTOTUNE)
        self.data_train = train

        valid = raw_valid.map(self.tf_encode)
        valid = valid.filter(
            lambda pt, en: tf.size(pt) <= max_len
            and tf.size(en) <= max_len
        )
        valid = valid.padded_batch(
            batch_size,
            padded_shapes=([None], [None]),
            padding_values=(
                tf.constant(0, tf.int64),
                tf.constant(0, tf.int64)
            )
        )
        self.data_valid = valid

    def encode(self, pt, en):
        """
        Tokenize and wrap with start/end tokens.
        """
        pt_text = pt.numpy().decode("utf-8")
        en_text = en.numpy().decode("utf-8")

        pt_ids = self.tokenizer_pt.encode(
            pt_text, add_special_tokens=False
        )
        en_ids = self.tokenizer_en.encode(
            en_text, add_special_tokens=False
        )

        return (
            [self.start_token] + pt_ids + [self.end_token],
            [self.start_token] + en_ids + [self.end_token]
        )

    def tf_encode(self, pt, en):
        """
        TF wrapper for encode().
        """
        pt_out, en_out = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_out.set_shape([None])
        en_out.set_shape([None])

        return pt_out, en_out
