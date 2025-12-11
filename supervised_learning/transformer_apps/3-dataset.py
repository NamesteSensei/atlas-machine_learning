#!/usr/bin/env python3
"""
Dataset class for tokenizing and batching the TED HRLR dataset.

Task 3:
- Add tf_encode() wrapper
- Filter by max_len
- Add caching, shuffling, padded batching, and prefetching
"""

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    Loads and prepares the TED HRLR dataset for machine translation.

    Attributes:
        data_train: Tokenized + batched training dataset.
        data_valid: Tokenized + batched validation dataset.
        tokenizer_pt: Portuguese tokenizer.
        tokenizer_en: English tokenizer.
        start_token: Start-of-sentence token index.
        end_token: End-of-sentence token index.
    """

    def __init__(self, batch_size, max_len):
        """
        Initialize dataset, tokenizers, and full TF pipeline.

        Args:
            batch_size (int): Size of padded batches.
            max_len (int): Maximum allowed token length.
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

        padding_i64 = tf.constant(0, dtype=tf.int64)

        train = train.padded_batch(
            batch_size,
            padded_shapes=([None], [None]),
            padding_values=(padding_i64, padding_i64)
        )

        train = train.prefetch(tf.data.experimental.AUTOTUNE)

        valid = valid.padded_batch(
            batch_size,
            padded_shapes=([None], [None]),
            padding_values=(padding_i64, padding_i64)
        )

        self.data_train = train
        self.data_valid = valid

    def encode(self, pt, en):
        """
        Python-side tokenization.

        Args:
            pt (tf.Tensor): Portuguese text.
            en (tf.Tensor): English text.

        Returns:
            tuple: Lists of token IDs.
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

        Returns:
            tuple: Two int64 tensors with defined shapes.
        """
        pt_out, en_out = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_out.set_shape([None])
        en_out.set_shape([None])

        return pt_out, en_out
