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