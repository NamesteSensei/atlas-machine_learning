#!/usr/bin/env python3
"""
Create padding and look-ahead masks for Transformer training.
"""

import tensorflow as tf


def create_padding_mask(seq):
    """
    Create a padding mask for a batch of token sequences.

    Args:
        seq (tf.Tensor): shape (batch_size, seq_len).

    Returns:
        tf.Tensor: shape (batch_size, 1, 1, seq_len) with 1. for pad tokens,
        0. elsewhere.
    """
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    mask = mask[:, tf.newaxis, tf.newaxis, :]
    return mask


def create_look_ahead_mask(size):
    """
    Create a look-ahead mask for self-attention.

    Args:
        size (tf.Tensor or int): sequence length.

    Returns:
        tf.Tensor: shape (size, size) with 1. above the diagonal (future),
        0. on and below it.
    """
    ones = tf.ones((size, size), dtype=tf.float32)
    band = tf.linalg.band_part(ones, -1, 0)
    mask = 1.0 - band
    return mask


def create_masks(inputs, target):
    """
    Create encoder, combined, and decoder masks for training.

    Args:
        inputs (tf.Tensor): shape (batch_size, seq_len_in).
        target (tf.Tensor): shape (batch_size, seq_len_out).

    Returns:
        tuple: (encoder_mask, combined_mask, decoder_mask)
    """
    encoder_mask = create_padding_mask(inputs)

    dec_target_padding_mask = create_padding_mask(target)

    seq_len_out = tf.shape(target)[1]
    look_ahead = create_look_ahead_mask(seq_len_out)
    look_ahead = look_ahead[tf.newaxis, tf.newaxis, :, :]

    combined_mask = tf.maximum(look_ahead, dec_target_padding_mask)

    decoder_mask = encoder_mask

    return encoder_mask, combined_mask, decoder_mask
