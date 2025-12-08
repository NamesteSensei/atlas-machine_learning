#!/usr/bin/env python3
"""Dataset class with batching and filtering"""

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Loads translation dataset and builds pipeline"""

    def __init__(self, batch_size, max_len):
        """Initializes dataset, tokenizers, and pipeline"""
        self.tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        self.tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

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

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        self.data_train = self.data_train.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )
        )
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(
            batch_size, padded_shapes=([None], [None])
        )
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        self.data_valid = self.data_valid.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )
        )
        self.data_valid = self.data_valid.padded_batch(
            batch_size, padded_shapes=([None], [None])
        )

    def encode(self, pt, en):
        """Encodes Portuguese and English sentences"""
        pt_tokens = self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8'),
            add_special_tokens=True
        )
        en_tokens = self.tokenizer_en.encode(
            en.numpy().decode('utf-8'),
            add_special_tokens=True
        )

        return (
            tf.convert_to_tensor(pt_tokens, dtype=tf.int64),
            tf.convert_to_tensor(en_tokens, dtype=tf.int64)
        )

    def tf_encode(self, pt, en):
        """TensorFlow wrapper for encode"""
        pt_ids, en_ids = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64]
        )
        pt_ids.set_shape([None])
        en_ids.set_shape([None])
        return pt_ids, en_ids
