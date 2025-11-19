#!/usr/bin/env python3
"""Calculate cumulative n-gram BLEU score"""

from collections import Counter
import math


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score.

    Args:
        references (list): list of reference translations
        sentence (list): candidate translation
        n (int): max n-gram size

    Returns:
        float: BLEU score
    """

    def ngrams(seq, n):
        """Generate n-grams from a sequence"""
        return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]

    precisions = []
    for i in range(1, n + 1):
        sent_ngrams = Counter(ngrams(sentence, i))
        max_ref_ngrams = {}

        for ref in references:
            ref_ngrams = Counter(ngrams(ref, i))
            for ng in ref_ngrams:
                max_ref_ngrams[ng] = max(
                    max_ref_ngrams.get(ng, 0), ref_ngrams[ng]
                )

        match = 0
        total = sum(sent_ngrams.values())
        for ng in sent_ngrams:
            match += min(sent_ngrams[ng], max_ref_ngrams.get(ng, 0))

        precision = match / total if total > 0 else 0
        precisions.append(precision)

    ref_lens = [len(ref) for ref in references]
    ref_len = min(ref_lens, key=lambda r: abs(r - len(sentence)))

    if len(sentence) > ref_len:
        bp = 1
    elif len(sentence) == 0:
        bp = 0
    else:
        bp = math.exp(1 - (ref_len / len(sentence)))

    score = 1
    for p in precisions:
        if p == 0:
            return 0
        score *= p ** (1 / n)

    return bp * score
