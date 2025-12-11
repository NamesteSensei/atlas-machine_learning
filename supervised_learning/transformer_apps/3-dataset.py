#!/usr/bin/env python3
"""
Dataset pipeline for TED HRLR Portuguese–English translation.

Task 3:
- Add filtering using max_len
- Add caching, shuffling, padded batching, and prefetching
- Maintain allowed imports only (tfds, transformers, tf)
"""

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    Builds a full TensorFlow data pipeline for machine translation.

    Attributes:
        data_train: Training dataset (batched, padded, shuffled).
        data_valid: Validation dataset (batched, padded).
        tokenizer_pt: Portuguese tokenizer.
        tokenizer_en: English tokenizer.
        start_token: Shared start token index.
        end_token: Shared end token index.
    """

    def __init__(self, batch_size, max_len):
        """
        Initialize dataset, tokenizers, and full data pipeline.

        Args:
            batch_size (int): Batch size for training and validation.
            max_len (int): Maximum allowed token length per sequence.
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
            lambda pt, en: tf.size(pt) <= max_len and tf.size(en) <= max_len
        )
        train = train.cache()
        train = train.shuffle(20000)

        train = train.padded_batch(
            batch_size,
            padded_shapes=([None], [None]),
            padding_values=(
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int64)
            )
        )

        train = train.prefetch(tf.data.experimental.AUTOTUNE)
        self.data_train = train

        valid = raw_valid.map(self.tf_encode)
        valid = valid.filter(
            lambda pt, en: tf.size(pt) <= max_len and tf.size(en) <= max_len
        )

        valid = valid.padded_batch(
            batch_size,
            padded_shapes=([None], [None]),
            padding_values=(
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int64)
            )
        )

        self.data_valid = valid

    def encode(self, pt, en):
        """
        Encode PT and EN sentences into token lists.

        Args:
            pt (tf.Tensor): Portuguese text.
            en (tf.Tensor): English text.

        Returns:
            tuple: (pt_tokens, en_tokens) as Python lists.
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
            pt (tf.Tensor): Portuguese input string.
            en (tf.Tensor): English input string.

        Returns:
            tuple: (pt_tensor, en_tensor) as int64 tensors.
        """
        pt_out, en_out = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            tout=[tf.int64, tf.int64]
        )

        pt_out.set_shape([None])
        en_out.set_shape([None])

        return pt_out, en_out
