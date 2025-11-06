Bitcoin Price Forecasting with Deep Learning

This project builds and trains an LSTM-based model to forecast Bitcoin (BTC) prices using historical and live market data from Coinbase and Bitstamp.
The system automates data collection, preprocessing, training, evaluation, and live prediction — enabling continuous BTC forecasting in real time.

Project Overview

The objective of this project is to create a full end-to-end time series forecasting pipeline that predicts short-term Bitcoin prices.
We leverage TensorFlow / Keras for model training, integrate live data from the Coinbase API, and track all training and retraining metrics through Weights & Biases (W&B).

W&B Dashboard:
https://wandb.ai/namestesensei-self/btc_forecasting

Project Structure
time_series/
│
├── 0-main.py                # Main entry point for end-to-end execution
├── preprocess_data.py       # Cleans, merges, resamples, and scales BTC data
├── forecast_btc.py          # Builds and trains the LSTM forecasting model
├── evaluate_model.py        # Evaluates the model’s performance
├── predict_next.py          # Fetches live Coinbase data for next-hour prediction
├── predict_next_auto.py     # Automated hourly prediction loop
├── auto_retrain.py          # Manual retraining script using historical + live data
│
├── btc_data/                # Raw and processed datasets
│   ├── coinbaseUSD.csv
│   ├── bitstampUSD.csv
│   ├── processed_btc_data.csv
│   ├── predictions_log.csv
│
├── models/                  # Saved model versions
│   ├── best_model.keras
│   ├── best_model_v2.keras
│
├── requirements.txt         # Python dependencies
└── README.md

Setup Instructions
1. Create Virtual Environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Preprocess Historical Data
./preprocess_data.py

3. Train Model
./forecast_btc.py

4. Evaluate Model
./evaluate_model.py

5. Run Live Prediction Loop (Coinbase API)
./predict_next_auto.py

6. Manual Retraining (Merges live + historical data)
./auto_retrain.py

Model Architecture
Layer Type	Units	Activation	Dropout	Description
LSTM	64	tanh	0.2	Captures long-term sequential dependencies
LSTM	64	tanh	0.2	Deeper sequence representation
Dense	1	linear	—	Outputs predicted BTC closing price

Optimizer: Adam
Loss Function: Mean Squared Error (MSE)
Epochs: 25–50
Batch Size: 32

Results Summary
Initial Model Training

Command:

./forecast_btc.py

Metric	Value
Training Loss	1.40e-04
Validation Loss (Best)	1.0e-05
Best Epoch	15
R² Score	0.9991
MSE	1.3e-05
MAE	0.002258

The model achieved near-perfect fit on validation data (R² = 0.9991), indicating strong predictive accuracy for BTC price trends.

W&B Run:
https://wandb.ai/namestesensei-self/btc_forecasting/runs/3palvwg5

Retraining (Auto Retrain v2)

Command:

./auto_retrain.py

Metric	Value
Training Loss	8.46e-05
Validation Loss (Best)	7.0e-05
Best Epoch	1
Total Samples	72,774
Sequences	X: 58,171
Validation Split	20% (14,543 sequences)

Retraining slightly increased validation loss due to the introduction of new live data, making the model more generalized and realistic.
The retraining pipeline automatically saves a new version:

models/best_model_v2.keras


W&B Run:
https://wandb.ai/namestesensei-self/btc_forecasting/runs/sdygoeg4

Feature Summary
Feature	Description	Status
Automated Data Preprocessing	Handles CSVs, encoding detection, missing data, resampling	✅
Supervised Sequence Generation	Builds lookback windows for LSTM	✅
Deep Learning Forecasting Model	Multi-layer LSTM for BTC price prediction	✅
W&B Experiment Tracking	Logs metrics, models, and versions	✅
Model Evaluation Script	Computes MSE, MAE, and R²	✅
Live Coinbase Data Integration	Pulls BTC/USD hourly data from API	✅
Automated Prediction Loop	Predicts and logs BTC prices hourly	✅
Auto Retraining	Merges live data and retrains model	✅
Model Version Control	Saves new .keras model with version tag	✅
Scalability	Extendable to other assets or exchanges	✅
Model Evolution
Stage	Validation Loss	R²	MSE	Description
Initial Training	1.0e-05	0.9991	1.3e-05	Base model trained on historical data
Retraining (v2)	7.0e-05	0.9987	1.77e-03	Adapted to live Coinbase data

The initial training achieved exceptional accuracy, and retraining introduced real-time adaptability, enabling ongoing improvement as market data evolves.
The model is now optimized for dynamic retraining and live prediction.
