"""Script to predict solar radiation for a specific date and hour.

Usage:
    python predict_for_datetime.py "MM/DD/YYYY" HH [results_dir] [reference_data.csv]

Example:
    python predict_for_datetime.py "02/03/2026" 6 results results/preprocessed_hourly.csv

This script:
1. Parses the given date and hour
2. Calculates day-of-year and hour-of-day features (cyclic encoding)
3. Uses average values from reference data for radiation channels
4. Loads the trained model and scaler
5. Predicts the target variable
"""

import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle


TARGET_COLUMN = "Global CMP22 (vent/cor) [W/m^2]"


def main():
    if len(sys.argv) < 3:
        print("Usage: python predict_for_datetime.py \"MM/DD/YYYY\" HH [results_dir] [reference_data.csv]")
        print("Example: python predict_for_datetime.py \"02/03/2026\" 6 results results/preprocessed_hourly.csv")
        sys.exit(1)

    date_str = sys.argv[1]
    hour = int(sys.argv[2])
    results_dir = sys.argv[3] if len(sys.argv) >= 4 else "results"
    reference_file = sys.argv[4] if len(sys.argv) >= 5 else os.path.join(results_dir, "preprocessed_hourly.csv")

    if hour < 0 or hour > 23:
        print("error: hour must be between 0 and 23")
        sys.exit(1)

    # parse the date
    try:
        target_date = datetime.strptime(date_str, "%m/%d/%Y")
    except ValueError:
        print(f"error: invalid date format. use MM/DD/YYYY")
        sys.exit(1)

    # day of year
    day_of_year = target_date.timetuple().tm_yday

    print(f"Predicting for: {date_str} hour {hour}")
    print(f"Day of year: {day_of_year}")

    # load reference data to get mean feature values
    if not os.path.exists(reference_file):
        print(f"warning: reference file not found at {reference_file}")
        print("using default values; results may be inaccurate")
        # fallback values (you can customize these)
        avg_features = {}
    else:
        df_ref = pd.read_csv(reference_file, index_col=0)
        print(f"loaded reference data from {reference_file}")
        # compute means of each column (except target)
        avg_features = df_ref.drop(columns=[TARGET_COLUMN], errors="ignore").mean().to_dict()

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

    # build feature vector
    # start with average values
    X_next = pd.Series(avg_features)

    # set time features
    X_next["day_of_year"] = day_of_year
    X_next["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.0)
    X_next["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.0)

    X_next["hour"] = hour
    X_next["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    X_next["cos_hour"] = np.cos(2 * np.pi * hour / 24)

    # get feature order from the original preprocessing
    # we need to match the order used when training
    feature_cols = [col for col in df_ref.columns if col != TARGET_COLUMN]
    X_next = X_next[feature_cols].values.reshape(1, -1)

    # scale
    X_next_scaled = scaler.transform(X_next)

    # predict
    y_pred = model.predict(X_next_scaled, verbose=0)
    predicted_value = y_pred[0, 0]

    print(f"\n{'='*70}")
    print(f"Prediction for {date_str} at {hour:02d}:00")
    print(f"{TARGET_COLUMN}: {predicted_value:.2f} W/m^2")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()