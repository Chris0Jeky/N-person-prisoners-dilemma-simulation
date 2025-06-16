#!/usr/bin/env python3
"""Quick test to debug the comparative analysis."""

import pandas as pd
import os

# Check if summary files exist
files_to_check = [
    'qlearning_results/1QL_experiments/csv/1QL_pairwise_cooperation_summary.csv',
    'qlearning_results/1QL_experiments/csv/1QL_pairwise_scores_summary.csv',
    'enhanced_qlearning_results/1EQL_experiments/csv/1EQL_pairwise_cooperation_summary.csv',
    'enhanced_qlearning_results/1EQL_experiments/csv/1EQL_pairwise_scores_summary.csv',
]

print("Checking summary files...")
for file in files_to_check:
    if os.path.exists(file):
        print(f"\n{file} exists")
        df = pd.read_csv(file)
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample experiments: {df['Experiment'].unique()[:3]}")
        print(f"Sample agents: {df['Agent'].unique()[:3]}")
    else:
        print(f"\n{file} NOT FOUND")

# Check detailed file naming
print("\n\nChecking detailed file naming...")
sample_files = [
    'qlearning_results/1QL_experiments/csv/1QL_pairwise_cooperation_1_QL_plus_2_AllD.csv',
    'enhanced_qlearning_results/1EQL_experiments/csv/1EQL_pairwise_cooperation_1_EQL_plus_2_AllD.csv'
]

for file in sample_files:
    print(f"\n{file}: {'EXISTS' if os.path.exists(file) else 'NOT FOUND'}")