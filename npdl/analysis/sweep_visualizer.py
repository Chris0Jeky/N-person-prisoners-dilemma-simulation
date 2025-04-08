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

    if target_coop not in df.columns:
        print(f"Warning: Primary metric '{target_coop}' not found in {csv_file}. Skipping some plots.")
        target_coop = None  # Disable plots relying on it

    # --- Generate Plots ---

    # 1. Pair Plot (Good overview, can be slow/dense for many params)
    try:
        if target_coop and len(param_cols) > 1:
            print("  Generating pair plot...")
            pair_plot_vars = param_cols + [target_coop]
            # Limit number of parameters for readability if too many
            if len(pair_plot_vars) > 6:
                print(f"    (Limiting pair plot to first {6 - 1} params + coop rate)")
                pair_plot_vars = param_cols[:(6 - 1)] + [target_coop]

            # Select a param with few unique values for hue if possible
            hue_param = None
            for p in param_cols:
                if df[p].nunique() < 6:  # Example threshold for hue categories
                    hue_param = p
                    break

    except Exception as e:
        print(f"  Error generating pair plot: {e}")