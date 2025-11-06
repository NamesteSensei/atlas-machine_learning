#!/usr/bin/env python3
"""
Preprocess BTC data for LSTM forecasting (v3.0)
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import chardet

DATA_DIR = "btc_data"
SEQ_LENGTH = 60  # past hours to predict next
OUTPUT_X = os.path.join(DATA_DIR, "X.npy")
OUTPUT_Y = os.path.join(DATA_DIR, "y.npy")


def detect_encoding(filepath):
    """Detect file encoding."""
    with open(filepath, "rb") as f:
        raw = f.read(10000)
    result = chardet.detect(raw)
    return result["encoding"]


def load_and_merge():
    """Load, clean, and merge Coinbase + Bitstamp BTC data."""
    files = ["coinbaseUSD.csv", "bitstampUSD.csv"]
    dfs = []

    for fname in files:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"⚠️ File not found: {fname}, skipping.")
            continue

        encoding = detect_encoding(fpath)
        print(f">>> Reading {fname} (encoding={encoding})")

        df = pd.read_csv(fpath, encoding=encoding, on_bad_lines="skip")
        if "Timestamp" not in df.columns or "Close" not in df.columns:
            raise ValueError(f"{fname} missing required columns")

        # Drop NaN close values
        df = df.dropna(subset=["Close"])
        df = df[df["Close"] > 0]

        # Convert UNIX timestamp to datetime
        df["date"] = pd.to_datetime(df["Timestamp"], unit="s", errors="coerce")
        df = df[["date", "Close"]].rename(columns={"Close": "close"})
        dfs.append(df)

    if not dfs:
        raise ValueError("❌ No valid CSVs loaded in btc_data/")

    # Merge both exchanges and clean
    df = pd.concat(dfs, axis=0)
    df = df.drop_duplicates(subset="date").sort_values("date")
    df.set_index("date", inplace=True)

    # Resample hourly and fill gaps
    df = df.resample("1h").mean().ffill()

    print(f"✅ Loaded {len(df)} rows ({df.index.min()} → {df.index.max()})")
    print(f"Sample:\n{df.head()}")
    return df


def scale_and_sequence(df):
    """Scale close prices and create time sequences."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["scaled_close"] = scaler.fit_transform(df[["close"]])

    X, y = [], []
    values = df["scaled_close"].values
    for i in range(len(values) - SEQ_LENGTH):
        X.append(values[i:i + SEQ_LENGTH])
        y.append(values[i + SEQ_LENGTH])

    X, y = np.array(X), np.array(y)
    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)

    print(f"✅ Created sequences: X={X.shape}, y={y.shape}")
    return X, y


def main():
    print(">>> Starting preprocessing...")
    df = load_and_merge()
    X, y = scale_and_sequence(df)
    print(">>> Preprocessing complete. Data saved to btc_data/.")


if __name__ == "__main__":
    main()
