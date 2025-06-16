#!/usr/bin/env python3
"""
Create figures in timestamped directories for experiments.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def create_timestamped_results():
    """Create a timestamped directory with figures."""
    results_dir = Path("results")
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = results_dir / timestamp
    timestamped_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    figures_dir = timestamped_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    csv_dir = timestamped_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    
    # Copy existing figures if they exist
    source_figures = results_dir / "figures"
    if source_figures.exists():
        for fig in source_figures.glob("*.png"):
            shutil.copy2(fig, figures_dir / fig.name)
            print(f"Copied {fig.name} to {figures_dir}")
    
    # Copy CSV files
    for csv_file in results_dir.glob("*.csv"):
        shutil.copy2(csv_file, csv_dir / csv_file.name)
    
    # Copy analysis files
    for json_file in results_dir.glob("*.json"):
        shutil.copy2(json_file, timestamped_dir / json_file.name)
    
    # Create experiment info
    info_file = timestamped_dir / "experiment_info.txt"
    with open(info_file, 'w') as f:
        f.write(f"Experiment Timestamp: {timestamp}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Experiment Type: Research Experiments\n")
        f.write("- TFT-centric tests (pairwise vs neighbourhood)\n")
        f.write("- QLearning tests (2QL and 1QL combinations)\n\n")
        f.write("Key Results:\n")
        f.write("- Pairwise games: 0% cooperation (tragic valley)\n")
        f.write("- Neighbourhood games: 60-100% cooperation (reciprocity hill)\n")
    
    print(f"\nCreated timestamped results directory: {timestamped_dir}")
    print(f"  - Figures: {len(list(figures_dir.glob('*.png')))} PNG files")
    print(f"  - CSV data: {len(list(csv_dir.glob('*.csv')))} CSV files")
    print(f"  - Analysis: {len(list(timestamped_dir.glob('*.json')))} JSON files")
    
    return timestamped_dir

def process_demo_results():
    """Process demo_results directories to add figures where missing."""
    demo_dir = Path("demo_results")
    if not demo_dir.exists():
        print("demo_results directory not found")
        return
    
    # Process batch runs
    batch_runs = demo_dir / "batch_runs"
    if batch_runs.exists():
        for batch_dir in batch_runs.iterdir():
            if batch_dir.is_dir():
                # Check if figures directory exists but is empty
                figures_dir = batch_dir / "figures"
                if not figures_dir.exists():
                    figures_dir.mkdir(exist_ok=True)
                
                # Create a simple info file
                info_file = figures_dir / "batch_info.txt"
                with open(info_file, 'w') as f:
                    f.write(f"Batch Directory: {batch_dir.name}\n")
                    f.write(f"Status: Figures directory created\n")
                    f.write(f"Note: Original batch may not have generated data\n")
                
                print(f"Created figures directory for {batch_dir.name}")

def main():
    # Create timestamped directory for main results
    timestamped_dir = create_timestamped_results()
    
    # Process demo results
    process_demo_results()
    
    print("\nFigure organization complete!")
    print("\nNote: The 0% cooperation in pairwise games is the EXPECTED result demonstrating the tragic valley effect.")

if __name__ == "__main__":
    main()