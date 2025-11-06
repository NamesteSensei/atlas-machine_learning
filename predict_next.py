#!/usr/bin/env python3
"""
predict_next.py
--------------------------------
Fetches live Bitcoin price data from Coinbase and uses the trained LSTM model
to predict the next 24 hours of BTC/USD prices.
"""

import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ===========================
# CONFIGURATION
# ===========================
MODEL_PATH = "models/best_model.keras"
SEQ_LEN = 60  # last 60 hours used to predict next one
FUTURE_HOURS = 24  # predict next 24 hours
COINBASE_API = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
SAVE_PATH = "btc_data/predictions_log.csv"

# ===========================
# FETCH LIVE DATA
# ===========================
def fetch_live_data():
    print(">>> Fetching live BTC/USD hourly data from Coinbase...")
    response = requests.get(COINBASE_API)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch Coinbase data: {response.status_code}")

    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"✅ Retrieved {len(df)} rows of live data (hourly candles).")
    return df

# ===========================
# PREPROCESS DATA
# ===========================
def preprocess_live_data(df):
    print(">>> Preprocessing live data for model input...")
    df = df[["close"]].astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X_input = []
    for i in range(len(scaled_data) - SEQ_LEN):
        X_input.append(scaled_data[i:i + SEQ_LEN])

    X_input = np.array(X_input)
    print(f"✅ Created input sequence: {X_input.shape}")
    return X_input, scaler, df

# ===========================
# PREDICT FUTURE (FIXED)
# ===========================
def predict_future(model, data, scaler):
    print(f">>> Predicting next {FUTURE_HOURS} hours...")
    last_sequence = data[-1]  # shape (60, 1)
    predictions = []

    # Reshape to 3D for model input
    current_seq = last_sequence.reshape(1, SEQ_LEN, 1)

    for _ in range(FUTURE_HOURS):
        next_pred = model.predict(current_seq, verbose=0)  # shape (1, 1)
        next_value = float(next_pred.squeeze())  # convert to scalar
        predictions.append(next_value)

        # Append new value properly (keep same 3D shape)
        next_value_array = np.array(next_value).reshape(1, 1, 1)
        current_seq = np.concatenate((current_seq[:, 1:, :], next_value_array), axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    print("✅ Predictions complete.")
    return predicted_prices

# ===========================
# PLOT RESULTS
# ===========================
def plot_predictions(df, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"][-200:], df["close"].values[-200:], label="Historical BTC Price")
    future_timestamps = pd.date_range(df["timestamp"].iloc[-1], periods=FUTURE_HOURS + 1, freq="H")[1:]
    plt.plot(future_timestamps, predictions, label="Predicted BTC Price", linestyle="dashed", color="orange")
    plt.title("BTC/USD Price Forecast (Next 24 Hours)")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ===========================
# SAVE PREDICTIONS
# ===========================
def save_predictions(predictions, df):
    future_timestamps = pd.date_range(df["timestamp"].iloc[-1], periods=FUTURE_HOURS + 1, freq="H")[1:]
    pred_df = pd.DataFrame({"timestamp": future_timestamps, "predicted_price": predictions.flatten()})

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    if os.path.exists(SAVE_PATH):
        pred_df.to_csv(SAVE_PATH, mode="a", header=False, index=False)
    else:
        pred_df.to_csv(SAVE_PATH, index=False)

    print(f"📝 Predictions saved to {SAVE_PATH}")

# ===========================
# MAIN
# ===========================
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first with forecast_btc.py.")

    model = load_model(MODEL_PATH)
    df_live = fetch_live_data()
    X_input, scaler, df = preprocess_live_data(df_live)
    predictions = predict_future(model, X_input, scaler)
    save_predictions(predictions, df_live)
    plot_predictions(df_live, predictions)

if __name__ == "__main__":
    main()
