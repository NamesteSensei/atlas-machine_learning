#!/usr/bin/env python3
"""Calculate n-gram BLEU score"""

from collections import Counter
import math


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    Args:
        references (list): list of reference translations
        sentence (list): candidate translation
        n (int): size of n-grams to evaluate

    Returns:
        float: n-gram BLEU score
    """

    def ngrams(seq, n):
        """Generate n-grams from sequence"""
        return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]

    sentence_ngrams = Counter(ngrams(sentence, n))

    max_ref_ngrams = {}
    for ref in references:
        ref_ngrams = Counter(ngrams(ref, n))
        for ng in ref_ngrams:
            max_ref_ngrams[ng] = max(
                max_ref_ngrams.get(ng, 0), ref_ngrams[ng]
            )

    clipped_count = 0
    total_count = sum(sentence_ngrams.values())

    for ng in sentence_ngrams:
        match = min(sentence_ngrams[ng], max_ref_ngrams.get(ng, 0))
        clipped_count += match

    precision = clipped_count / total_count if total_count > 0 else 0

    ref_lens = [len(ref) for ref in references]
    ref_len = min(ref_lens, key=lambda r: abs(r - len(sentence)))

    if len(sentence) > ref_len:
        bp = 1
    else:
        if len(sentence) == 0:
            bp = 0
        else:
            bp = math.exp(1 - (ref_len / len(sentence)))

    return bp * precision
