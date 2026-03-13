"""Utility to visualize attribute relationships (correlation heatmap).

Usage:
    python visualize_attributes.py data.csv

The script reads a CSV file with a datetime index or separate date/time columns,
computes the pairwise Pearson correlation of numeric features and writes a
heatmap image (`attribute_correlation.png`) to the current directory.
"""

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
    elif "DATE (MM/DD/YYYY)" in df.columns and "MST" in df.columns:
        df["datetime"] = pd.to_datetime(df["DATE (MM/DD/YYYY)"] + " " + df["MST"])
        df.set_index("datetime", inplace=True)
        df = df.drop(["DATE (MM/DD/YYYY)", "MST"], axis=1)
    else:
        try:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df.set_index(df.columns[0], inplace=True)
        except Exception:
            pass
    return df


def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_attributes.py data.csv")
        sys.exit(1)

    df = load_csv(sys.argv[1])
    # drop non-numeric columns
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Attribute Correlation Matrix")
    plt.tight_layout()
    plt.savefig("attribute_correlation.png")
    print("Saved attribute_correlation.png")


if __name__ == "__main__":
    main()