"""Script to predict the next hour's solar radiation using the trained ANN model.

Usage:
    python predict_next_hour.py [preprocessed_data.csv] [results_dir]

Example:
    python predict_next_hour.py results/preprocessed_hourly.csv results

This script:
1. Loads the last row from the preprocessed data
2. Generates time features for the next hour
3. Loads the trained model and scaler
4. Makes a prediction for the next hour's target variable
"""

import sys
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle


TARGET_COLUMN = "Global CMP22 (vent/cor) [W/m^2]"


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_next_hour.py [preprocessed_data.csv] [results_dir]")
        sys.exit(1)

    data_file = sys.argv[1]
    results_dir = sys.argv[2] if len(sys.argv) >= 3 else "results"

    # load preprocessed data
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    latest_row = df.iloc[-1].copy()
    latest_time = df.index[-1]
    next_time = latest_time + timedelta(hours=1)

    print(f"Latest data timestamp: {latest_time}")
    print(f"Predicting for: {next_time}")

    # load model and scaler
    model_path = os.path.join(results_dir, "model.h5")
    scaler_path = os.path.join(results_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        print(f"error: model file not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"error: scaler file not found at {scaler_path}")
        sys.exit(1)

    model = tf.keras.models.load_model(model_path, compile=False)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # prepare input for next hour
    # drop the target column and any other non-feature columns if needed
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]
    
    # create input from latest row, update time features for next hour
    X_next = latest_row[feature_cols].values.reshape(1, -1)

    # update time features if they exist
    if "hour" in feature_cols:
        hour_idx = feature_cols.index("hour")
        next_hour = next_time.hour
        X_next[0, hour_idx] = next_hour

    if "sin_hour" in feature_cols:
        sin_idx = feature_cols.index("sin_hour")
        X_next[0, sin_idx] = np.sin(2 * np.pi * next_time.hour / 24)

    if "cos_hour" in feature_cols:
        cos_idx = feature_cols.index("cos_hour")
        X_next[0, cos_idx] = np.cos(2 * np.pi * next_time.hour / 24)

    if f"{TARGET_COLUMN}_lag1" in feature_cols:
        lag_idx = feature_cols.index(f"{TARGET_COLUMN}_lag1")
        X_next[0, lag_idx] = latest_row[TARGET_COLUMN]

    # scale using the same scaler as training
    X_next_scaled = scaler.transform(X_next)

    # predict
    y_pred = model.predict(X_next_scaled, verbose=0)
    predicted_value = y_pred[0, 0]

    print(f"\n{'='*60}")
    print(f"Prediction for {next_time}:")
    print(f"{TARGET_COLUMN}: {predicted_value:.2f} W/m^2")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()