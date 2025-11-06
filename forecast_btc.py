#!/usr/bin/env python3
"""
Train LSTM model for BTC price forecasting (v3.0)
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import wandb
from wandb.integration.keras import WandbCallback

DATA_DIR = "btc_data"
MODEL_DIR = "models"
SEQ_LENGTH = 60
EPOCHS = 50
BATCH_SIZE = 64


def load_data():
    """Load preprocessed numpy sequences."""
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    if X.size == 0 or y.size == 0:
        raise ValueError("❌ Empty training data! Run preprocess_data.py first.")
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    return X_train, X_val, y_train, y_val


def build_model():
    """Build stacked LSTM."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    wandb.init(project="btc_forecasting", config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQ_LENGTH,
        "architecture": "Stacked LSTM (128→64) + Dropout 0.2"
    })

    print(">>> Loading data...")
    X_train, X_val, y_train, y_val = load_data()

    model = build_model()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.keras"),
                        save_best_only=True, monitor="val_loss"),
        WandbCallback(save_model=False)
    ]

    print(">>> Training model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    model.save(os.path.join(MODEL_DIR, "final_model.keras"))
    print("✅ Training complete! Model saved in models/")
    wandb.finish()


if __name__ == "__main__":
    main()
