#!/usr/bin/env python3
"""
Diagnose the actual data to understand why pairwise shows 0 cooperation.
"""

import csv
import json
from pathlib import Path

def check_csv_data():
    """Check what's actually in the CSV files."""
    results_dir = Path("results")
    
    print("Checking pairwise TFT cooperation data...")
    print("=" * 60)
    
    # Check 3 TFT pairwise data
    pairwise_file = results_dir / "pairwise_tft_cooperation_3_TFT_aggregated.csv"
    if pairwise_file.exists():
        with open(pairwise_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            print(f"File: {pairwise_file.name}")
            print(f"Total rows: {len(rows)}")
            
            # Show first 5 and last 5 rows
            print("\nFirst 5 rows:")
            for i, row in enumerate(rows[:5]):
                print(f"Round {row['Round']}: Mean={row['Mean_Cooperation_Rate']}, "
                      f"Std={row['Std_Dev']}, CI=[{row['Lower_95_CI']}, {row['Upper_95_CI']}]")
            
            print("\nLast 5 rows:")
            for row in rows[-5:]:
                print(f"Round {row['Round']}: Mean={row['Mean_Cooperation_Rate']}, "
                      f"Std={row['Std_Dev']}, CI=[{row['Lower_95_CI']}, {row['Upper_95_CI']}]")
    
    print("\n" + "=" * 60)
    print("Checking neighbourhood TFT cooperation data...")
    
    # Check 3 TFT neighbourhood data
    neighbourhood_file = results_dir / "neighbourhood_tft_cooperation_3_TFT_aggregated.csv"
    if neighbourhood_file.exists():
        with open(neighbourhood_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            print(f"File: {neighbourhood_file.name}")
            print(f"Total rows: {len(rows)}")
            
            # Show first 5 and last 5 rows
            print("\nFirst 5 rows:")
            for i, row in enumerate(rows[:5]):
                if i < len(rows):
                    print(f"Round {row.get('Round', 'N/A')}: Mean={row.get('Mean_Cooperation_Rate', 'N/A')}")
    
    # Check comparative analysis
    print("\n" + "=" * 60)
    print("Checking comparative analysis...")
    
    comp_file = results_dir / "comparative_analysis.json"
    if comp_file.exists():
        with open(comp_file, 'r') as f:
            data = json.load(f)
            
        tft_analysis = data.get('tft_analysis', {})
        
        print("\nPairwise final cooperation rates:")
        for config, stats in tft_analysis.get('pairwise', {}).items():
            print(f"  {config}: {stats.get('mean_final_cooperation', 'N/A')}")
        
        print("\nNeighbourhood final cooperation rates:")
        for config, stats in tft_analysis.get('neighbourhood', {}).items():
            print(f"  {config}: {stats.get('mean_final_cooperation', 'N/A')}")
    
    # Check if there's actual history data
    print("\n" + "=" * 60)
    print("Checking for history files...")
    
    history_files = list(results_dir.glob("*_history.csv"))
    print(f"Found {len(history_files)} history files")
    
    if history_files:
        # Check one history file
        sample_file = history_files[0]
        print(f"\nSampling {sample_file.name}:")
        
        with open(sample_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if rows:
                print(f"Columns: {list(rows[0].keys())}")
                print(f"First row: {rows[0]}")

if __name__ == "__main__":
    check_csv_data()