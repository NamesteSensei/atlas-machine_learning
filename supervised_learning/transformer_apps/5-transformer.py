#!/usr/bin/env python3
"""
Complete Transformer model
"""
import tensorflow as tf
# Ensure these are implemented in your previous tasks
Encoder = __import__('2-transformer').Encoder
Decoder = __import__('3-transformer').Decoder


class Transformer(tf.keras.Model):
    """
    Transformer model for machine translation
    """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_len):
        """
        Init for Transformer
        """
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_len)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_len)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        Forward pass for the Transformer
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)
        final_output = self.linear(dec_output)

        return final_output
