#!/usr/bin/env python3
"""
Training script for the Transformer model.
"""
import tensorflow as tf

# Import your existing Task 3 and Task 4 logic
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warmup."""
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Trains a transformer model for machine translation."""
    data = Dataset(batch_size, max_len)
    input_v = data.tokenizer_pt.vocab_size + 2
    target_v = data.tokenizer_en.vocab_size + 2

    model = Transformer(N, dm, h, hidden, input_v, target_v, max_len)
    optimizer = tf.keras.optimizers.Adam(CustomSchedule(dm), beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_fn(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_obj(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        return tf.reduce_sum(loss_ * mask) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

    @tf.function
    def train_step(inp, tar):
        tar_inp, tar_real = tar[:, :-1], tar[:, 1:]
        enc_m, comb_m, dec_m = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            pred = model(inp, tar_inp, True, enc_m, comb_m, dec_m)
            loss = loss_fn(tar_real, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)
        train_acc(tar_real, pred)

    for epoch in range(epochs):
        train_loss.reset_state()
        train_acc.reset_state()
        for (batch, (inp, tar)) in enumerate(data.data_train):
            train_step(inp, tar)
            if batch % 50 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch}: Loss "
                      f"{train_loss.result()} Accuracy {train_acc.result()}")
        print(f"Epoch {epoch + 1}: Loss {train_loss.result()} "
              f"Accuracy {train_acc.result()}")
    return model
