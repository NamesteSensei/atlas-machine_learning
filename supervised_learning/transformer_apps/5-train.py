#!/usr/bin/env python3
"""
Trains a Transformer model for Portuguese to English translation.
"""

import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate schedule with warmup.
    """

    def __init__(self, dm, warmup_steps=4000):
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        Computes the learning rate for a given training step.
        """
        # ✅ FIX: step must be float for rsqrt
        step = tf.cast(step, tf.float32)

        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """
    Computes masked sparse categorical crossentropy loss.
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        real, pred, from_logits=True
    )
    mask = tf.cast(mask, loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    """
    Computes masked accuracy.
    """
    predictions = tf.argmax(pred, axis=2)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    matches = tf.equal(real, predictions)
    matches = tf.logical_and(mask, matches)
    return tf.reduce_sum(tf.cast(matches, tf.float32)) / tf.reduce_sum(
        tf.cast(mask, tf.float32)
    )


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a Transformer model.
    """
    dataset = Dataset(batch_size, max_len)

    input_vocab = dataset.tokenizer_pt.vocab_size + 2
    target_vocab = dataset.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N, dm, h, hidden,
        input_vocab, target_vocab, max_len
    )

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_accuracy = 0.0

        for batch, (inputs, target) in enumerate(dataset.data_train):
            target_in = target[:, :-1]
            target_real = target[:, 1:]

            encoder_mask, combined_mask, decoder_mask = create_masks(
                inputs, target_in
            )

            with tf.GradientTape() as tape:
                predictions = transformer(
                    inputs,
                    target_in,
                    True,
                    encoder_mask,
                    combined_mask,
                    decoder_mask
                )
                loss = loss_function(target_real, predictions)
                accuracy = accuracy_function(target_real, predictions)

            gradients = tape.gradient(
                loss, transformer.trainable_variables
            )
            optimizer.apply_gradients(
                zip(gradients, transformer.trainable_variables)
            )

            total_loss += loss
            total_accuracy += accuracy

            if batch % 50 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch}: "
                    f"Loss {loss} Accuracy {accuracy}"
                )

        print(
            f"Epoch {epoch}: "
            f"Loss {total_loss / (batch + 1)} "
            f"Accuracy {total_accuracy / (batch + 1)}"
        )

    return transformer
