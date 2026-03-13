"""A simple end-to-end pipeline for training an ANN on the
z4689499 dataset.  

Usage examples:
    python pipeline_ann.py z4689499.csv

The script will:
* read & parse the CSV
* resample to hourly slots (mean of values for that hour)
* optionally drop redundant attributes by correlation
* split into train/validation/test sets
* standardize features
* build a Keras sequential model
* train the model and plot the training history
* evaluate on the test set and visualize predictions

By default the target column is the first numeric column after the
datetime fields ("Global CMP22 (vent/cor) [W/m^2]") but you can
change TARGET_COLUMN at the top of the file.
"""

import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras

# configuration
TARGET_COLUMN = "Global CMP22 (vent/cor) [W/m^2]"
CORRELATION_THRESHOLD = 0.98  # drop perfectly / highly correlated features
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # if a datetime column already exists, use it
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        return df

    # combine date and time columns if necessary
    if "DATE (MM/DD/YYYY)" in df.columns and "MST" in df.columns:
        df["datetime"] = pd.to_datetime(df["DATE (MM/DD/YYYY)"] + " " + df["MST"])
        df = df.drop(["DATE (MM/DD/YYYY)", "MST"], axis=1)
    else:
        # try to parse first column as datetime and drop it
        try:
            df.index = pd.to_datetime(df.iloc[:, 0])
            df = df.iloc[:, 1:]
        except Exception:
            pass

    # ensure final index is named 'datetime' for consistency
    df.index.name = "datetime"
    return df


def preprocess(df: pd.DataFrame, out_dir: str = ".") -> pd.DataFrame:
    """Resample to hourly slots, engineer features and drop redundant columns.
    
    Also saves the preprocessed hourly dataframe to preprocessed_hourly.csv.
    """
    # hourly mean (pandas wants lowercase 'h' in some environments)
    df_hourly = df.resample("h").mean()
    df_hourly = df_hourly.dropna(how="any")

    # ---------- feature engineering ----------
    # day of year (0-364) with cyclic encoding – captures seasonal variation
    df_hourly["day_of_year"] = df_hourly.index.dayofyear
    df_hourly["sin_doy"] = np.sin(2 * np.pi * df_hourly["day_of_year"] / 365.0)
    df_hourly["cos_doy"] = np.cos(2 * np.pi * df_hourly["day_of_year"] / 365.0)

    # hour of day (0-23) with cyclic encoding – captures daily cycle
    df_hourly["hour"] = df_hourly.index.hour
    df_hourly["sin_hour"] = np.sin(2 * np.pi * df_hourly["hour"] / 24)
    df_hourly["cos_hour"] = np.cos(2 * np.pi * df_hourly["hour"] / 24)

    # simple lag of the target as an additional input (helps if forecasting)
    if TARGET_COLUMN in df_hourly.columns:
        df_hourly[f"{TARGET_COLUMN}_lag1"] = df_hourly[TARGET_COLUMN].shift(1)
        df_hourly = df_hourly.dropna(how="any")  # drop first row created by shift

    # ---------- end feature engineering ----------

    # drop highly correlated columns (simple heuristic)
    corr = df_hourly.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > CORRELATION_THRESHOLD)]
    if to_drop:
        print(f"dropping columns due to high correlation: {to_drop}")
        df_hourly = df_hourly.drop(columns=to_drop)

    # save the preprocessed hourly dataframe
    save_path = os.path.join(out_dir, "preprocessed_hourly.csv")
    df_hourly.to_csv(save_path)
    print(f"saved preprocessed hourly data to {save_path}")

    return df_hourly


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1)  # regression output
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def plot_history(history: keras.callbacks.History, out_dir: str):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history.get("val_loss", []), label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("Training history")
    plt.savefig(os.path.join(out_dir, "training_history.png"))
    plt.close()


def plot_predictions(y_true, y_pred, index, out_dir: str, name: str):
    plt.figure(figsize=(10, 4))
    plt.plot(index, y_true, label="actual")
    plt.plot(index, y_pred, label="predicted")
    plt.xlabel("Time")
    plt.ylabel(TARGET_COLUMN)
    plt.legend()
    plt.title(f"{name} predictions")
    fname = os.path.join(out_dir, f"{name}_predictions.png")
    plt.savefig(fname)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline_ann.py data.csv [output_dir]")
        sys.exit(1)

    datafile = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else "results"
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(datafile)
    print(f"loaded {len(df)} rows, columns: {list(df.columns)}")

    df = preprocess(df, out_dir=out_dir)
    print(f"after preprocessing, shape {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"target column {TARGET_COLUMN} not in data")

    X = df.drop(columns=[TARGET_COLUMN]).values
    y = df[TARGET_COLUMN].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = build_model(X_train.shape[1])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=2,
    )

    plot_history(history, out_dir)

    # evaluate
    test_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, test_pred)
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_pred)
    print(f"test MSE {mse:.4f}, RMSE {rmse:.4f}, MAE {mae:.4f}, R2 {r2:.4f}")

    # plot a slice of the test set
    plot_predictions(y_test, test_pred, df.index[-len(y_test):], out_dir, name="test")

    # save scaler and model
    model.save(os.path.join(out_dir, "model.h5"))
    pd.to_pickle(scaler, os.path.join(out_dir, "scaler.pkl"))
    print(f"outputs are in {out_dir}")


if __name__ == "__main__":
    main()
