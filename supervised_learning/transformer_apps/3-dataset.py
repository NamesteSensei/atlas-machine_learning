#!/usr/bin/env python3
"""
Dataset class for Portuguese to English translation
"""

import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """Loads and prepares the translation dataset"""

    def __init__(self):
        """Initialize dataset splits and tokenizers"""

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

        # 🔑 THESE TWO LINES WERE MISSING (ROOT CAUSE)
        self.input_vocab_size = self.tokenizer_pt.vocab_size
        self.target_vocab_size = self.tokenizer_en.vocab_size

    def tokenize_dataset(self, data):
        """Create pretrained tokenizers"""

        tokenizer_pt = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )

        tokenizer_en = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        return tokenizer_pt, tokenizer_en
