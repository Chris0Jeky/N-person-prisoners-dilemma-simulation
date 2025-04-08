import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def visualize_sweep_results(csv_file, output_dir):
    """Loads sweep results and generates plots."""

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        return
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return

    if df.empty:
        print(f"Warning: CSV file {csv_file} is empty.")
        return

    print(f"Visualizing results from: {csv_file}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Identify parameters and metrics ---
    metric_cols = [col for col in df.columns if col.startswith('avg_') or col.startswith('std_')]
    param_cols = [col for col in df.columns if col not in metric_cols]

    # Target metrics for main plots
    target_coop = 'avg_final_coop_rate_overall'
    target_score = 'avg_avg_final_score_target'  # Adjusted based on CSV headers