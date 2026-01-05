#!/usr/bin/env python3
"""
Train Transformer model for Portuguese to English translation.
"""

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

Dataset = __import__("3-dataset").Dataset
create_masks = __import__("4-create_masks").create_masks
Transformer = __import__("5-transformer").Transformer


class CustomSchedule(LearningRateSchedule):
    """Learning rate schedule with warmup."""

    def __init__(self, dm, warmup_steps=4000):
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction="none",
)


def loss_function(y_true, y_pred):
    """Masked sparse categorical crossentropy loss."""
    mask = tf.math.not_equal(y_true, 0)
    loss = loss_object(y_true, y_pred)
    mask = tf.cast(mask, loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def accuracy_function(y_true, y_pred):
    """Masked token-level accuracy."""
    y_pred = tf.argmax(y_pred, axis=2, output_type=tf.int64)
    matches = tf.equal(y_true, y_pred)
    mask = tf.math.not_equal(y_true, 0)
    matches = tf.cast(matches & mask, tf.float32)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_sum(matches) / tf.reduce_sum(mask)


def evaluate(model, dataset):
    """Evaluate model on validation dataset."""
    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    for inp, tar in dataset:
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_mask, look_mask, dec_mask = create_masks(
            inp,
            tar_inp,
        )

        preds = model(
            (inp, tar_inp),
            training=False,
            encoder_mask=enc_mask,
            look_ahead_mask=look_mask,
            decoder_mask=dec_mask,
        )

        total_loss += loss_function(tar_real, preds)
        total_acc += accuracy_function(tar_real, preds)
        batches += 1

    return total_loss / batches, total_acc / batches


def train_transformer(
    N,
    dm,
    h,
    hidden,
    max_len,
    batch_size,
    epochs,
):
    """Build, train, evaluate, and save Transformer model."""
    dataset = Dataset(batch_size, max_len)

    model = Transformer(
        N,
        dm,
        h,
        hidden,
        dataset.input_vocab_size,
        dataset.target_vocab_size,
        max_len,
    )

    lr = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        lr,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
    )

    for epoch in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        batches = 0

        for batch, (inp, tar) in enumerate(dataset.data_train):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_mask, look_mask, dec_mask = create_masks(
                inp,
                tar_inp,
            )

            with tf.GradientTape() as tape:
                preds = model(
                    (inp, tar_inp),
                    training=True,
                    encoder_mask=enc_mask,
                    look_ahead_mask=look_mask,
                    decoder_mask=dec_mask,
                )
                loss = loss_function(tar_real, preds)
                acc = accuracy_function(tar_real, preds)

            grads = tape.gradient(
                loss,
                model.trainable_variables,
            )
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables)
            )

            total_loss += loss
            total_acc += acc
            batches += 1

            if batch % 50 == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch}: "
                    f"Loss {loss:.4f}, Accuracy {acc:.4f}"
                )

        val_loss, val_acc = evaluate(
            model,
            dataset.data_valid,
        )

        print(
            f"Epoch {epoch + 1}: "
            f"Loss {total_loss / batches:.4f}, "
            f"Accuracy {total_acc / batches:.4f}, "
            f"Val Loss {val_loss:.4f}, "
            f"Val Accuracy {val_acc:.4f}"
        )

    model.save_weights("transformer_weights.h5")
    print("Saved model weights.")

    model.save("transformer_model")
    print("Saved full model.")

    return model
