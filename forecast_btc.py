#!/usr/bin/env python3
"""
forecast_btc.py
Train an LSTM model to forecast Bitcoin prices using cleaned data.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import wandb
from wandb.integration.keras import WandbCallback

# Initialize Weights & Biases
wandb.init(
    project="btc_forecasting",
    entity="namestesensei-self",
    name="forecast_btc_model"
)

DATA_PATH = "btc_data/processed_btc_data.csv"
MODEL_SAVE_PATH = "models/best_model.keras"


def load_data():
    """Load and scale preprocessed BTC data"""
    print(">>> Loading processed data")
    df = pd.read_csv(DATA_PATH)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    X = np.load("btc_data/X.npy")
    y = np.load("btc_data/y.npy")

    return X, y, scaler


def build_model(input_shape):
    """Define the LSTM model"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    print(">>> Loading preprocessed sequences...")
    X, y, scaler = load_data()
    print(f"Loaded: X={X.shape}, y={y.shape}")

    print(">>> Building model...")
    model = build_model((X.shape[1], X.shape[2]))

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_loss",
        save_best_only=True
    )

    print(">>> Starting training...")
    history = model.fit(
        X,
        y,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[
            early_stop,
            checkpoint,
            WandbCallback(save_model=False)
        ],
        verbose=1
    )

    print("✅ Training complete! Model saved in 'models/' directory.")


if __name__ == "__main__":
    main()
