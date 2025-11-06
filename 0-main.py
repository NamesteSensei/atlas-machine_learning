#!/usr/bin/env python3
"""
Quick test runner for preprocessing + model training
"""

import os

print(">>> Starting Preprocessing")
os.system("python3 preprocess_data.py")

print(">>> Starting Forecast Training")
os.system("python3 forecast_btc.py")
