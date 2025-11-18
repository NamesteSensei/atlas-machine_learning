#!/usr/bin/env python3
"""Calculate unigram BLEU score"""

from collections import Counter
import math


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    Args:
        references (list): list of reference translations,
                           each a list of words
        sentence (list): candidate translation (list of words)

    Returns:
        float: unigram BLEU score
    """
    # Count unigrams in sentence
    sentence_counter = Counter(sentence)

    # Build max reference counts for each word
    max_ref_counts = {}
    for ref in references:
        ref_counter = Counter(ref)
        for word in ref_counter:
            max_ref_counts[word] = max(
                max_ref_counts.get(word, 0), ref_counter[word]
            )

    # Count clipped matches
    clipped_count = 0
    total_count = sum(sentence_counter.values())

    for word in sentence_counter:
        match = min(sentence_counter[word], max_ref_counts.get(word, 0))
        clipped_count += match

    # Precision = matched unigrams / total unigrams
    precision = clipped_count / total_count if total_count > 0 else 0

    # Brevity penalty
    ref_lens = [len(ref) for ref in references]
    ref_len = min(ref_lens, key=lambda ref: abs(ref - len(sentence)))
    if len(sentence) > ref_len:
        bp = 1
    else:
        bp = math.exp(1 - (ref_len / len(sentence))) if len(sentence) > 0 else 0

    bleu = bp * precision
    return bleu
