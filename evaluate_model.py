#!/usr/bin/env python3
"""
Evaluate trained BTC forecasting model (v3.0)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import wandb

DATA_DIR = "btc_data"
MODEL_PATH = "models/best_model.keras"


def main():
    wandb.init(project="btc_forecasting", job_type="evaluation")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ No trained model found! Run forecast_btc.py first.")

    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    split = int(0.8 * len(X))
    X_test, y_test = X[split:], y[split:]
    X_test = np.expand_dims(X_test, axis=-1)

    model = load_model(MODEL_PATH)
    preds = model.predict(X_test).flatten()

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"📈 MSE: {mse:.6f}")
    print(f"📉 MAE: {mae:.6f}")
    print(f"🔢 R²:  {r2:.4f}")

    wandb.log({"mse": mse, "mae": mae, "r2": r2})

    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:500], label="Actual", alpha=0.7)
    plt.plot(preds[:500], label="Predicted", alpha=0.7)
    plt.title("BTC Forecast — Actual vs Predicted (Scaled)")
    plt.xlabel("Time Steps")
    plt.ylabel("Scaled Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    wandb.finish()


if __name__ == "__main__":
    main()
