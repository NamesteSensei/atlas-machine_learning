#!/usr/bin/env python3
"""
auto_retrain.py — Automated model retraining and W&B logging
Retrains your BTC forecasting model using both historical and live data.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import wandb
from wandb.integration.keras import WandbCallback

# --- W&B setup ---
wandb.init(
    project="btc_forecasting",
    name="auto_retrain_v2",
    config={
        "epochs": 30,
        "batch_size": 64,
        "sequence_length": 60,
        "architecture": "LSTM_2x64_dropout_0.2",
    },
)

# --- File paths ---
HISTORICAL_PATH = "btc_data/processed_btc_data.csv"
LIVE_LOG_PATH = "btc_data/predictions_log.csv"
MODEL_SAVE_PATH = "models/best_model_v2.keras"

# --- Load and merge data ---
def load_combined_data():
    print("📂 Loading historical and live BTC data...")

    if not os.path.exists(HISTORICAL_PATH):
        raise FileNotFoundError(f"Missing file: {HISTORICAL_PATH}")

    df_hist = pd.read_csv(HISTORICAL_PATH)
    print(f"✅ Loaded historical data: {df_hist.shape}")

    # Keep only relevant column
    if "close" not in df_hist.columns:
        print("⚠️ No 'close' column found — using last numeric column as close.")
        df_hist["close"] = df_hist.select_dtypes(include="number").iloc[:, -1]

    df_hist = df_hist[["close"]].dropna()

    # Load live data if it exists
    if os.path.exists(LIVE_LOG_PATH):
        df_live = pd.read_csv(LIVE_LOG_PATH)
        print(f"✅ Loaded live data: {df_live.shape}")

        # Rename prediction to close if necessary
        if "prediction" in df_live.columns:
            df_live.rename(columns={"prediction": "close"}, inplace=True)

        # Drop any nulls and keep only the close column
        if "close" in df_live.columns:
            df_live = df_live[["close"]].dropna()
        else:
            print("⚠️ Live log missing 'close' or 'prediction' column — skipping.")
            df_live = pd.DataFrame(columns=["close"])

        # Append live data to the historical
        combined = pd.concat([df_hist, df_live], axis=0, ignore_index=True)
    else:
        print("⚠️ No live log found, using historical data only.")
        combined = df_hist

    combined = combined.dropna().reset_index(drop=True)
    print(f"📈 Final merged dataset shape: {combined.shape}")

    return combined

# --- Create sequences ---
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# --- Build model ---
def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Main retraining logic ---
def main():
    print("🚀 Starting automated retraining process...")

    df = load_combined_data()
    if df.empty:
        raise ValueError("❌ No data available for retraining. Check data paths or formats.")

    # Prepare target values
    values = df["close"].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)

    # Create sequences
    seq_len = wandb.config.sequence_length
    X, y = create_sequences(scaled_data, seq_len)

    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    print(f"🧩 Training shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"🧪 Validation shapes: X={X_val.shape}, y={y_val.shape}")

    # Build model
    model = build_model((seq_len, 1))

    # Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # --- Train ---
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=wandb.config.epochs,
        batch_size=wandb.config.batch_size,
        callbacks=[checkpoint, early_stop, WandbCallback()],
        verbose=1
    )

    print(f"✅ Retraining complete! Model saved as {MODEL_SAVE_PATH}")
    wandb.finish()


if __name__ == "__main__":
    main()
