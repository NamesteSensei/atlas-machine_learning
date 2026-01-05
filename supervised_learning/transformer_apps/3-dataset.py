#!/usr/bin/env python3
"""Dataset loader for Portuguese-English translation."""

import tensorflow_datasets as tfds
from transformers import AutoTokenizer
import tensorflow as tf


class Dataset:
    """Loads and prepares dataset and tokenizers."""

    def __init__(self, batch_size, max_len):
        """Initialize dataset and tokenizers."""
        examples, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            with_info=True,
            as_supervised=True
        )
        train_examples = examples['train']
        val_examples = examples['validation']

        self.tokenizer_pt = AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        self.tokenizer_en = AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def encode(pt, en):
            pt_ids = self.tokenizer_pt.encode(
                pt.numpy().decode('utf-8'),
                truncation=True,
                padding='max_length',
                max_length=max_len
            )
            en_ids = self.tokenizer_en.encode(
                en.numpy().decode('utf-8'),
                truncation=True,
                padding='max_length',
                max_length=max_len
            )
            return pt_ids, en_ids

        def tf_encode(pt, en):
            result_pt, result_en = tf.py_function(
                func=encode,
                inp=[pt, en],
                Tout=[tf.int64, tf.int64]
            )
            result_pt.set_shape([max_len])
            result_en.set_shape([max_len])
            return result_pt, result_en

        train = train_examples.map(tf_encode)
        val = val_examples.map(tf_encode)

        self.data_train = train.cache().shuffle(20000).padded_batch(
            batch_size, padded_shapes=([max_len], [max_len])
        ).prefetch(tf.data.experimental.AUTOTUNE)

        self.data_valid = val.padded_batch(
            batch_size, padded_shapes=([max_len], [max_len])
        )
