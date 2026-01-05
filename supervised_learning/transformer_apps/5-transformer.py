#!/usr/bin/env python3
"""
Transformer model for machine translation (Portuguese to English).
"""

import tensorflow as tf


def scaled_dot_product_attention(Q, K, V, mask):
    """
    Calculate the attention weights.

    Args:
        Q, K, V: query, key, value matrices
        mask: mask tensor

    Returns:
        output, attention_weights
    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer."""
    def __init__(self, dm, h):
        super().__init__()
        assert dm % h == 0
        self.dm = dm
        self.h = h
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        batch_size = tf.shape(Q)[0]

        Q = self.split_heads(self.Wq(Q), batch_size)
        K = self.split_heads(self.Wk(K), batch_size)
        V = self.split_heads(self.Wv(V), batch_size)

        scaled_attention, weights = scaled_dot_product_attention(Q, K, V, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dm))

        return self.linear(concat_attention), weights


class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding for input sequences."""
    def __init__(self, position, dm):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, dm)

    def get_angles(self, pos, i, dm):
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(dm, tf.float32))
        return pos * angle_rates

    def positional_encoding(self, position, dm):
        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(dm, dtype=tf.float32)[tf.newaxis, :],
            dm
        )

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


class EncoderBlock(tf.keras.layers.Layer):
    """Single encoder block."""
    def __init__(self, dm, h, hidden, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(dm),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """Single decoder block."""
    def __init__(self, dm, h, hidden, dropout):
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(dm),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        out1 = self.layernorm1(x + self.dropout1(attn1, training=training))

        attn2, _ = self.mha2(out1, enc_output, enc_output, padding_mask)
        out2 = self.layernorm2(out1 + self.dropout2(attn2, training=training))

        ffn_output = self.ffn(out2)
        out3 = self.layernorm3(out2 + self.dropout3(ffn_output, training=training))
        return out3, attn1, attn2


class Encoder(tf.keras.layers.Layer):
    """Encoder composed of multiple EncoderBlocks."""
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, dropout=0.1):
        super().__init__()
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.pos_encoding = PositionalEncoding(max_seq_len, dm)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.blocks = [EncoderBlock(dm, h, hidden, dropout) for _ in range(N)]

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    """Decoder composed of multiple DecoderBlocks."""
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, dropout=0.1):
        super().__init__()
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.pos_encoding = PositionalEncoding(max_seq_len, dm)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.blocks = [DecoderBlock(dm, h, hidden, dropout) for _ in range(N)]

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x, _, _ = block(x, enc_output, training, look_ahead_mask, padding_mask)
        return x


class Transformer(tf.keras.Model):
    """Full Transformer model."""
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab, max_seq_len, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_len, dropout)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_len, dropout)
        self.final_layer = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, training, encoder_mask, look_ahead_mask, decoder_mask):
        inp, tar = inputs
        enc_output = self.encoder(inp, training, encoder_mask)
        dec_output = self.decoder(tar, enc_output, training, look_ahead_mask, decoder_mask)
        return self.final_layer(dec_output)
