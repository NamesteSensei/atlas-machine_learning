#!/usr/bin/env python3
"""
Defines a complete Transformer model for machine translation.

This module implements the Transformer architecture described in
"Attention Is All You Need", including:
- Positional encoding
- Scaled dot-product attention
- Multi-head attention
- Encoder and decoder stacks
"""

import tensorflow as tf


def positional_encoding(max_len, dm):
    """
    Creates positional encoding for input sequences.

    Args:
        max_len (int): Maximum sequence length
        dm (int): Model dimensionality

    Returns:
        tf.Tensor: Positional encoding of shape (1, max_len, dm)
    """
    positions = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
    dims = tf.range(dm, dtype=tf.float32)[tf.newaxis, :]

    angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / dm)
    angle_rads = positions * angle_rates

    sines = tf.sin(angle_rads[:, 0::2])
    cosines = tf.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    return pos_encoding[tf.newaxis, ...]


def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculates scaled dot-product attention.

    Args:
        q (tf.Tensor): Query tensor
        k (tf.Tensor): Key tensor
        v (tf.Tensor): Value tensor
        mask (tf.Tensor or None): Attention mask

    Returns:
        tf.Tensor: Attention output
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_logits, axis=-1)
    return tf.matmul(attention_weights, v)


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention layer.
    """

    def __init__(self, dm, h):
        """
        Initializes the multi-head attention layer.

        Args:
            dm (int): Model dimensionality
            h (int): Number of attention heads
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.wq = tf.keras.layers.Dense(dm)
        self.wk = tf.keras.layers.Dense(dm)
        self.wv = tf.keras.layers.Dense(dm)
        self.dense = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Splits the last dimension into (h, depth).

        Args:
            x (tf.Tensor): Input tensor
            batch_size (int): Batch size

        Returns:
            tf.Tensor: Reshaped tensor
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        """
        Applies multi-head attention.

        Args:
            q (tf.Tensor): Query
            k (tf.Tensor): Key
            v (tf.Tensor): Value
            mask (tf.Tensor): Mask tensor

        Returns:
            tf.Tensor: Output tensor
        """
        batch_size = tf.shape(q)[0]

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        attention = scaled_dot_product_attention(q, k, v, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.dm)
        )
        return self.dense(concat_attention)


class EncoderBlock(tf.keras.layers.Layer):
    """
    Encoder block consisting of attention and feed-forward layers.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initializes the encoder block.

        Args:
            dm (int): Model dimensionality
            h (int): Number of heads
            hidden (int): Hidden layer size
            drop_rate (float): Dropout rate
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(dm)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Forward pass for the encoder block.
        """
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class DecoderBlock(tf.keras.layers.Layer):
    """
    Decoder block for the Transformer.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initializes the decoder block.
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(dm)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass for the decoder block.
        """
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(ffn_output + out2)


class Transformer(tf.keras.Model):
    """
    Full Transformer model for machine translation.
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_len):
        """
        Initializes the Transformer.

        Args:
            N (int): Number of encoder/decoder blocks
            dm (int): Model dimensionality
            h (int): Number of heads
            hidden (int): Hidden layer size
            input_vocab (int): Input vocabulary size
            target_vocab (int): Target vocabulary size
            max_len (int): Maximum sequence length
        """
        super().__init__()

        self.encoder_embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.decoder_embedding = tf.keras.layers.Embedding(target_vocab, dm)

        self.pos_encoding = positional_encoding(max_len, dm)

        self.enc_layers = [
            EncoderBlock(dm, h, hidden) for _ in range(N)
        ]
        self.dec_layers = [
            DecoderBlock(dm, h, hidden) for _ in range(N)
        ]

        self.final_layer = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training,
             encoder_mask, combined_mask, decoder_mask):
        """
        Forward pass for the Transformer.
        """
        seq_len_in = tf.shape(inputs)[1]
        seq_len_out = tf.shape(target)[1]

        enc_output = self.encoder_embedding(inputs)
        enc_output += self.pos_encoding[:, :seq_len_in, :]

        for layer in self.enc_layers:
            enc_output = layer(enc_output, training, encoder_mask)

        dec_output = self.decoder_embedding(target)
        dec_output += self.pos_encoding[:, :seq_len_out, :]

        for layer in self.dec_layers:
            dec_output = layer(
                dec_output, enc_output, training,
                combined_mask, decoder_mask
            )

        return self.final_layer(dec_output)
