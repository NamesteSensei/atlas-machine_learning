#!/usr/bin/env python3
"""
Module for loading the TED HRLR Portuguese-to-English translation dataset.

This module provides a function `load_dataset` that loads a specified
split of the TensorFlow Dataset 'ted_hrlr_translate/pt_to_en'. The returned
dataset contains (Portuguese_sentence, English_sentence) tensor pairs.

Allowed imports:
    - tensorflow_datasets as tfds
"""

import tensorflow_datasets as tfds
import transformers


def load_dataset(split="train"):
    """
    Load a split of the TED HRLR Portuguese-to-English translation dataset.

    Args:
        split (str): Dataset split to load. Common values include "train",
                     "validation", and "test".

    Returns:
        tf.data.Dataset: A dataset of `(pt, en)` sentence pairs.
    """
    return tfds.load(
        "ted_hrlr_translate/pt_to_en",
        split=split,
        as_supervised=True
    )
