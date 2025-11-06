#!/usr/bin/env python3
"""
predict_next_auto.py
-------------------------------------------------
Continuously fetches live BTC/USD price data from Coinbase
and uses your trained LSTM model to predict the next 24 hours
of Bitcoin prices every hour.

Press CTRL + C to stop.
"""

import os
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

MODEL_PATH = "models/best_model.keras"
SEQ_LEN = 60
FUTURE_HOURS = 24
COINBASE_API = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
SAVE_PATH = "btc_data/predictions_log.csv"

# -------------------------------
# Fetch and preprocess live data
# -------------------------------
def fetch_live_data():
    response = requests.get(COINBASE_API)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch Coinbase data: {response.status_code}")
    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def preprocess_live_data(df_live):
    df_close = df_live[["close"]].astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df_close)
    X_input = np.array([scaled[i:i + SEQ_LEN] for i in range(len(scaled) - SEQ_LEN)])
    return X_input, scaler

# -------------------------------
# Predict future prices
# -------------------------------
def predict_future(model, data, scaler):
    last_seq = data[-1].reshape(1, SEQ_LEN, 1)
    preds = []
    for _ in range(FUTURE_HOURS):
        next_pred = model.predict(last_seq, verbose=0)
        next_val = float(next_pred.squeeze())
        preds.append(next_val)
        next_val_array = np.array(next_val).reshape(1, 1, 1)
        last_seq = np.concatenate((last_seq[:, 1:, :], next_val_array), axis=1)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1))

# -------------------------------
# Save and plot results
# -------------------------------
def save_predictions(preds, df_live):
    future_times = pd.date_range(df_live["timestamp"].iloc[-1], periods=FUTURE_HOURS + 1, freq="h")[1:]
    pred_df = pd.DataFrame({"timestamp": future_times, "predicted_price": preds.flatten()})
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    pred_df.to_csv(SAVE_PATH, mode="a", header=not os.path.exists(SAVE_PATH), index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Predictions saved to {SAVE_PATH}")

def plot_predictions(df_live, preds):
    plt.figure(figsize=(12, 6))
    plt.plot(df_live["timestamp"][-200:], df_live["close"].values[-200:], label="Historical BTC Price")
    future_timestamps = pd.date_range(df_live["timestamp"].iloc[-1], periods=FUTURE_HOURS + 1, freq="h")[1:]
    plt.plot(future_timestamps, preds, label="Predicted BTC Price", linestyle="dashed", color="orange")
    plt.title(f"BTC/USD Forecast — Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()

# -------------------------------
# Continuous loop
# -------------------------------
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first.")

    model = load_model(MODEL_PATH)
    print("🚀 Starting live prediction loop (updates hourly). Press CTRL+C to stop.\n")

    while True:
        try:
            print(">>> Fetching latest BTC/USD data...")
            df_live = fetch_live_data()
            print(f"✅ Retrieved {len(df_live)} rows of live data.")

            X_input, scaler = preprocess_live_data(df_live)
            preds = predict_future(model, X_input, scaler)

            save_predictions(preds, df_live)
            plot_predictions(df_live, preds)

            print("⏳ Sleeping for 1 hour before next update...\n")
            time.sleep(3600)

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
            break
        except Exception as e:
            print(f"⚠️ Error occurred: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
