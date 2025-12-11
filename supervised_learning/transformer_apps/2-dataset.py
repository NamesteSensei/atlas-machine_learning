#!/usr/bin/env python3
"""
Dataset class for loading, encoding, and mapping TED HRLR Portuguese–English data.

Task 2: Add tf_encode() as a TensorFlow wrapper around encode(),
and update dataset splits to return tokenized tensors.
"""

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    Loads and prepares the TED HRLR dataset for machine translation.

    Attributes:
        data_train: Training dataset, tokenized.
        data_valid: Validation dataset, tokenized.
        tokenizer_pt: Portuguese tokenizer.
        tokenizer_en: English tokenizer.
        start_token: Shared start-of-sentence token index.
        end_token: Shared end-of-sentence token index.
    """

    def __init__(self):
        """Initialize dataset, tokenizers, and mapped encoding."""
        # Load raw datasets
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

        # Load tokenizers (shared vocabulary architecture)
        self.tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )
        self.tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )

        # Shared special tokens
        self.start_token = self.tokenizer_pt.vocab_size
        self.end_token = self.start_token + 1

        # Apply TensorFlow mapping
        self.data_train = raw_train.map(self.tf_encode)
        self.data_valid = raw_valid.map(self.tf_encode)

    def encode(self, pt, en):
        """
        Encodes Portuguese and English sentences into token lists.

        Args:
            pt (tf.Tensor): Portuguese sentence.
            en (tf.Tensor): English sentence.

        Returns:
            list: pt_tokens, en_tokens
        """
        pt_str = pt.numpy().decode("utf-8")
        en_str = en.numpy().decode("utf-8")

        pt_ids = self.tokenizer_pt.encode(pt_str, add_special_tokens=False)
        en_ids = self.tokenizer_en.encode(en_str, add_special_tokens=False)

        pt_tokens = [self.start_token] + pt_ids + [self.end_token]
        en_tokens = [self.start_token] + en_ids + [self.end_token]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode() method.

        Args:
            pt (tf.Tensor): Portuguese sentence.
            en (tf.Tensor): English sentence.

        Returns:
            tuple: (pt_tensor, en_tensor) with shapes set.
        """
        pt_encoded, en_encoded = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        # Set shapes so TensorFlow knows these are 1D variable-length tensors
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])

        return pt_encoded, en_encoded
#!/usr/bin/env python3
"""
Dataset class for loading, encoding, and mapping TED HRLR Portuguese–English data.

Task 2: Add tf_encode() as a TensorFlow wrapper around encode(),
and update dataset splits to return tokenized tensors.
"""

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    Loads and prepares the TED HRLR dataset for machine translation.

    Attributes:
        data_train: Training dataset, tokenized.
        data_valid: Validation dataset, tokenized.
        tokenizer_pt: Portuguese tokenizer.
        tokenizer_en: English tokenizer.
        start_token: Shared start-of-sentence token index.
        end_token: Shared end-of-sentence token index.
    """

    def __init__(self):
        """Initialize dataset, tokenizers, and mapped encoding."""
        # Load raw datasets
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

        # Load tokenizers (shared vocabulary architecture)
        self.tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )
        self.tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )

        # Shared special tokens
        self.start_token = self.tokenizer_pt.vocab_size
        self.end_token = self.start_token + 1

        # Apply TensorFlow mapping
        self.data_train = raw_train.map(self.tf_encode)
        self.data_valid = raw_valid.map(self.tf_encode)

    def encode(self, pt, en):
        """
        Encodes Portuguese and English sentences into token lists.

        Args:
            pt (tf.Tensor): Portuguese sentence.
            en (tf.Tensor): English sentence.

        Returns:
            list: pt_tokens, en_tokens
        """
        pt_str = pt.numpy().decode("utf-8")
        en_str = en.numpy().decode("utf-8")

        pt_ids = self.tokenizer_pt.encode(pt_str, add_special_tokens=False)
        en_ids = self.tokenizer_en.encode(en_str, add_special_tokens=False)

        pt_tokens = [self.start_token] + pt_ids + [self.end_token]
        en_tokens = [self.start_token] + en_ids + [self.end_token]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode() method.

        Args:
            pt (tf.Tensor): Portuguese sentence.
            en (tf.Tensor): English sentence.

        Returns:
            tuple: (pt_tensor, en_tensor) with shapes set.
        """
        pt_encoded, en_encoded = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        # Set shapes so TensorFlow knows these are 1D variable-length tensors
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])

        return pt_encoded, en_encoded
