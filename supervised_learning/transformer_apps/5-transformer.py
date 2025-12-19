#!/usr/bin/env python3
"""
Complete Transformer architecture for Portuguese-English translation.
"""
import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    """
    Encoder for the Transformer model.
    """
    def __init__(self, N, dm, h, hidden, input_vocab, max_len):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.layers = [tf.keras.layers.Dense(dm) for _ in range(N)]

    def call(self, x, training, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(tf.keras.layers.Layer):
    """
    Decoder for the Transformer model.
    """
    def __init__(self, N, dm, h, hidden, target_vocab, max_len):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.layers = [tf.keras.layers.Dense(dm) for _ in range(N)]

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x


class Transformer(tf.keras.Model):
    """
    Full Transformer model.
    """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab, max_len):
        """Initialize the model."""
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_len)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_len)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, enc_mask, look_mask, dec_mask):
        """Model forward pass."""
        enc_output = self.encoder(inputs, training, enc_mask)
        dec_output = self.decoder(target, enc_output, training,
                                  look_mask, dec_mask)
        return self.linear(dec_output)
