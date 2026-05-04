import pandas as pd
import numpy as np
import os

def calculate_variability(file_path, n_samples=1000):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Preprocessing
    # The file has DATE (MM/DD/YYYY) and MST columns
    df['datetime'] = pd.to_datetime(df['DATE (MM/DD/YYYY)'] + ' ' + df['MST'])
    df.set_index('datetime', inplace=True)
    col = 'Global CMP22 (vent/cor) [W/m^2]'
    
    # 1. Minute-Level Data (First 1000 daytime samples)
    # Daytime filter (Ghi > 10) to avoid zero-variance noise
    data_min_day = df[df[col] > 10][col]
    if len(data_min_day) < n_samples:
        print(f"Warning: Only {len(data_min_day)} minute samples available.")
        sample_min = data_min_day
    else:
        sample_min = data_min_day.iloc[:n_samples]
    
    # 2. Hourly-Level Data (First 1000 daytime samples)
    data_hourly = df[col].resample('h').mean()
    data_hourly_day = data_hourly[data_hourly > 10]
    if len(data_hourly_day) < n_samples:
        print(f"Warning: Only {len(data_hourly_day)} hourly samples available.")
        sample_hourly = data_hourly_day
    else:
        sample_hourly = data_hourly_day.iloc[:n_samples]
        
    # Calculate Differences (t_i - t_{i-1})
    diff_min = sample_min.diff().dropna()
    diff_hourly = sample_hourly.diff().dropna()
    
    # Metrics
    # Volatility = Std Dev of differences
    # Variance = Variance of differences
    metrics = {
        'Metric': ['Variance of Change', 'Volatility (Std Dev)', 'Mean Abs Change', 'Max Jump'],
        'Minute-Level (N=1000)': [
            diff_min.var(),
            diff_min.std(),
            diff_min.abs().mean(),
            diff_min.abs().max()
        ],
        'Hourly-Level (N=1000)': [
            diff_hourly.var(),
            diff_hourly.std(),
            diff_hourly.abs().mean(),
            diff_hourly.abs().max()
        ]
    }
    
    results = pd.DataFrame(metrics)
    # Use actual lengths in case data was smaller than n_samples
    n_min = len(diff_min)
    n_hou = len(diff_hourly)
    
    results['Ratio (Hourly/Minute)'] = results['Hourly-Level (N=1000)'] / results['Minute-Level (N=1000)']
    
    print(f"\nVariability Comparison (Step-wise Changes)")
    print(f"Minute Samples: {n_min+1} | Hourly Samples: {n_hou+1}")
    print("="*80)
    print(results.to_string(index=False))
    print("="*80)
    
    results.to_csv('variability_comparison_n1000.csv', index=False)
    print("\nResults saved to variability_comparison_n1000.csv")

if __name__ == "__main__":
    file = 'z4689499.csv'
    if os.path.exists(file):
        calculate_variability(file)
    else:
        # Check if it's in the current directory or parent
        if os.path.exists('c:/Users/likhi/OneDrive/Documents/btp_files/z4689499.csv'):
            calculate_variability('c:/Users/likhi/OneDrive/Documents/btp_files/z4689499.csv')
        else:
            print(f"File not found: {file}")
