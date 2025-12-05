#!/usr/bin/env python3
"""Dataset class with encode wrapper for Transformer translation"""
import tensorflow_datasets as tfds
import tensorflow as tf


class Dataset:
    """Loads and prepares dataset for Transformer model"""

    def __init__(self):
        """Initializes training and validation datasets and tokenizers"""
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

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Creates subword tokenizers for both Portuguese and English sentences

        Args:
            data: tf.data.Dataset, Portuguese-English sentence pairs

        Returns:
            tuple: (tokenizer_pt, tokenizer_en)
        """
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data),
            target_vocab_size=2**13
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data),
            target_vocab_size=2**13
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes Portuguese and English sentences into token IDs with SOS/EOS tokens

        Args:
            pt (tf.Tensor): Portuguese sentence
            en (tf.Tensor): English sentence

        Returns:
            tuple: (encoded_pt_tensor, encoded_en_tensor)
        """
        pt_tokens = self.tokenizer_pt.encode(pt.numpy())
        en_tokens = self.tokenizer_en.encode(en.numpy())

        pt_tokens = [self.tokenizer_pt.vocab_size] + pt_tokens + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + en_tokens + [self.tokenizer_en.vocab_size + 1]

        return tf.convert_to_tensor(pt_tokens, dtype=tf.int64), tf.convert_to_tensor(en_tokens, dtype=tf.int64)

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for encode() using tf.py_function

        Args:
            pt (tf.Tensor): Portuguese sentence
            en (tf.Tensor): English sentence

        Returns:
            tuple: (encoded_pt_tensor, encoded_en_tensor)
        """
        result_pt, result_en = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
