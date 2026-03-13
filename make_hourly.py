"""Generate an hourly-aggregated dataset from the original minute-level CSV.

Usage:
    python make_hourly.py input.csv output_hourly.csv

The output can then be fed directly into the ANN pipeline.
"""

import sys
import pandas as pd


def main():
    if len(sys.argv) != 3:
        print("Usage: python make_hourly.py input.csv output_hourly.csv")
        sys.exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]

    df = pd.read_csv(infile)
    if "DATE (MM/DD/YYYY)" in df.columns and "MST" in df.columns:
        df["datetime"] = pd.to_datetime(df["DATE (MM/DD/YYYY)"] + " " + df["MST"])
        df = df.drop(["DATE (MM/DD/YYYY)", "MST"], axis=1)
    else:
        # attempt to parse first column
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.rename(columns={df.columns[0]: "datetime"}, inplace=True)

    df.set_index("datetime", inplace=True)

    hourly = df.resample("h").mean().dropna(how="any")
    hourly.to_csv(outfile)
    print(f"Wrote {len(hourly)} hourly rows to {outfile}")


if __name__ == "__main__":
    main()