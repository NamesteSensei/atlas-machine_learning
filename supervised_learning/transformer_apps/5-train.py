#!/usr/bin/env python3
"""Train Transformer model on Portuguese-English dataset."""

import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
Transformer = __import__('5-transformer').Transformer
create_masks = __import__('4-create_masks').create_masks


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule with warmup."""

    def __init__(self, dm, warmup_steps=4000):
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def loss_function(y_true, y_pred):
    """Sparse categorical crossentropy ignoring padding."""
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    mask = tf.math.not_equal(y_true, 0)
    loss = loss_obj(y_true, y_pred)
    mask = tf.cast(mask, loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def accuracy_function(y_true, y_pred):
    """Token accuracy ignoring padding."""
    y_pred = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
    matches = tf.equal(y_true, y_pred)
    mask = tf.math.not_equal(y_true, 0)
    matches = tf.logical_and(matches, mask)
    matches = tf.cast(matches, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_sum(matches) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Train and return a Transformer model."""
    dataset = Dataset(batch_size, max_len)

    model = Transformer(
        N,
        dm,
        h,
        hidden,
        dataset.tokenizer_pt.vocab_size,
        dataset.tokenizer_en.vocab_size,
        max_len
    )

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")

        total_loss = 0.0
        total_acc = 0.0
        batches = 0

        for batch, (inp, tar) in enumerate(dataset.data_train):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_mask, look_ahead_mask, dec_mask = create_masks(
                inp,
                tar_inp
            )

            with tf.GradientTape() as tape:
                predictions = model(
                    (inp, tar_inp),
                    True,
                    enc_mask,
                    look_ahead_mask,
                    dec_mask
                )
                loss = loss_function(tar_real, predictions)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables)
            )

            acc = accuracy_function(tar_real, predictions)

            total_loss += loss
            total_acc += acc
            batches += 1

            if batch % 50 == 0:
                print(
                    f"Batch {batch}: "
                    f"Loss {loss:.4f}, "
                    f"Accuracy {acc:.4f}"
                )

        print(
            f"Epoch {epoch + 1}: "
            f"Loss {total_loss / batches:.4f}, "
            f"Accuracy {total_acc / batches:.4f}"
        )

    model.save("transformer_model")
    print("Model saved: transformer_model")

    return model
