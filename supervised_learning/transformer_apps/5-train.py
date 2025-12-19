#!/usr/bin/env python3
"""
Task 5: Train a Transformer model
"""

import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate schedule for the Transformer.
    """

    def __init__(self, d_model, warmup_steps=4000):
        """
        Initialize the learning rate schedule.

        Args:
            d_model (int): Model dimensionality
            warmup_steps (int): Warmup steps
        """
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        Compute learning rate.

        Args:
            step (tf.Tensor): Training step

        Returns:
            tf.Tensor: Learning rate
        """
        step = tf.cast(step, tf.float32)

        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def train_transformer(dataset, epochs):
    """
    Train a Transformer model.

    Args:
        dataset (Dataset): Dataset instance
        epochs (int): Number of epochs

    Returns:
        tf.keras.Model: Trained Transformer model
    """
    Transformer = __import__('4-transformer').Transformer

    transformer = Transformer(
        N=4,
        dm=128,
        h=8,
        hidden=512,
        input_vocab=dataset.tokenizer_pt.vocab_size,
        target_vocab=dataset.tokenizer_en.vocab_size,
        max_seq_input=dataset.max_len,
        max_seq_target=dataset.max_len
    )

    learning_rate = CustomSchedule(128)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )

    def loss_function(real, pred):
        """
        Compute masked loss.

        Args:
            real (tf.Tensor): True labels
            pred (tf.Tensor): Predictions

        Returns:
            tf.Tensor: Loss value
        """
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy'
    )

    @tf.function
    def train_step(inp, tar):
        """
        Perform one training step.
        """
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = (
            __import__('4-create_masks').create_masks(inp, tar_inp)
        )

        with tf.GradientTape() as tape:
            predictions = transformer(
                inp,
                tar_inp,
                training=True,  # ✅ REQUIRED FIX
                enc_padding_mask=enc_padding_mask,
                look_ahead_mask=combined_mask,
                dec_padding_mask=dec_padding_mask
            )

            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(
            loss,
            transformer.trainable_variables
        )

        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    for epoch in range(epochs):
        train_loss.reset_state()
        train_accuracy.reset_state()

        for batch, (inp, tar) in enumerate(dataset.data_train):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch}: "
                    f"Loss {train_loss.result()} "
                    f"Accuracy {train_accuracy.result()}"
                )

        print(
            f"Epoch {epoch + 1}: "
            f"Loss {train_loss.result()} "
            f"Accuracy {train_accuracy.result()}"
        )

    return transformer
