#!/usr/bin/env python3
"""
Create figures for experiments in timestamped directories.
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime


def create_simple_figure(data, output_path, title):
    """Create a simple text-based visualization."""
    with open(output_path, 'w') as f:
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n\n")
        
        if 'cooperation_rates' in data:
            f.write("Cooperation Rate Over Time:\n")
            rates = data['cooperation_rates']
            max_rate = max(rates) if rates else 1
            
            for i, rate in enumerate(rates[:20]):  # Show first 20 rounds
                bar_length = int(rate * 40 / max_rate) if max_rate > 0 else 0
                bar = "#" * bar_length
                f.write(f"Round {i+1:3d}: [{bar:<40}] {rate:.3f}\n")
            
            if len(rates) > 20:
                f.write(f"... (showing first 20 of {len(rates)} rounds)\n")
            
            f.write(f"\nFinal cooperation rate: {rates[-1] if rates else 0:.3f}\n")
        
        if 'agent_scores' in data:
            f.write("\n\nFinal Agent Scores:\n")
            for agent_id, score in data['agent_scores'].items():
                f.write(f"  Agent {agent_id}: {score}\n")
        
        f.write("\n" + "=" * 60 + "\n")


def process_experiment_directory(exp_dir):
    """Process a single experiment directory and create figures."""
    exp_path = Path(exp_dir)
    if not exp_path.exists() or not exp_path.is_dir():
        return False
    
    # Create figures directory
    figures_dir = exp_path / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Look for results files
    json_files = list(exp_path.glob("*_results.json"))
    
    if json_files:
        # Process JSON results
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract cooperation rates
            coop_rates = []
            if 'history' in data:
                for round_data in data['history']:
                    if 'cooperation_rate' in round_data:
                        coop_rates.append(round_data['cooperation_rate'])
            
            # Extract final scores
            agent_scores = data.get('final_scores', {})
            
            # Create simple visualization
            viz_data = {
                'cooperation_rates': coop_rates,
                'agent_scores': agent_scores
            }
            
            output_file = figures_dir / f"{json_file.stem}_visualization.txt"
            create_simple_figure(viz_data, output_file, f"Results for {json_file.stem}")
            
            print(f"Created visualization: {output_file}")
    
    # Look for CSV files
    csv_dir = exp_path / "csv"
    if csv_dir.exists():
        history_files = list(csv_dir.glob("*_history.csv"))
        
        for hist_file in history_files:
            # Read CSV data
            coop_data = []
            with open(hist_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'cooperated' in row:
                        coop_data.append(int(row['cooperated']))
            
            # Calculate cooperation rate per round
            if coop_data:
                # Assuming 3 agents per round
                rounds = len(coop_data) // 3
                coop_rates = []
                for i in range(rounds):
                    round_coop = sum(coop_data[i*3:(i+1)*3]) / 3.0
                    coop_rates.append(round_coop)
                
                viz_data = {'cooperation_rates': coop_rates}
                output_file = figures_dir / f"{hist_file.stem}_cooperation.txt"
                create_simple_figure(viz_data, output_file, f"Cooperation Analysis: {hist_file.stem}")
                
                print(f"Created CSV visualization: {output_file}")
    
    return True


def find_and_process_experiments(base_dir="demo_results"):
    """Find all experiment directories and process them."""
    base_path = Path(base_dir)
    processed = 0
    
    print("Searching for experiment directories...")
    
    # Process direct subdirectories
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            if process_experiment_directory(item):
                processed += 1
    
    # Process batch runs
    batch_dir = base_path / "batch_runs"
    if batch_dir.exists():
        for batch in batch_dir.iterdir():
            if batch.is_dir():
                for exp in batch.iterdir():
                    if exp.is_dir() and not exp.name == "figures":
                        if process_experiment_directory(exp):
                            processed += 1
    
    # Process basic runs
    basic_dir = base_path / "basic_runs"
    if basic_dir.exists():
        for exp in basic_dir.iterdir():
            if exp.is_dir():
                if process_experiment_directory(exp):
                    processed += 1
    
    # Process comparative analysis subdirectories
    comp_dir = base_path / "comparative_analysis"
    if comp_dir.exists():
        for item in comp_dir.iterdir():
            if item.is_dir() and item.name.startswith("scaling_test"):
                if process_experiment_directory(item):
                    processed += 1
    
    print(f"\nProcessed {processed} experiment directories")
    
    # Also process main results directory
    main_results = Path("results")
    if main_results.exists():
        print("\nProcessing main results directory...")
        
        # Create timestamp directory for current results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_dir = main_results / timestamp
        timestamp_dir.mkdir(exist_ok=True)
        
        figures_dir = timestamp_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Copy analysis results
        comp_analysis = main_results / "comparative_analysis.json"
        if comp_analysis.exists():
            with open(comp_analysis, 'r') as f:
                data = json.load(f)
            
            # Create summary visualization
            summary_file = figures_dir / "experiment_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("RESEARCH EXPERIMENT SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("TFT Analysis - Tragic Valley vs Reciprocity Hill:\n")
                f.write("-" * 40 + "\n")
                
                tft = data.get('tft_analysis', {})
                for config in ['3_TFT', '2_TFT-E__plus__1_AllD', '2_TFT__plus__1_AllD', '2_TFT-E__plus__1_AllC']:
                    if config in tft.get('differences', {}):
                        diff = tft['differences'][config]
                        f.write(f"\n{config}:\n")
                        f.write(f"  Pairwise cooperation: {tft['pairwise'][config]['mean_final_cooperation']:.3f}\n")
                        f.write(f"  Neighbourhood cooperation: {tft['neighbourhood'][config]['mean_final_cooperation']:.3f}\n")
                        f.write(f"  Difference: {diff['cooperation_difference']:.3f}\n")
                        f.write(f"  Effect: {'Reciprocity Hill' if diff['shows_reciprocity_hill'] else 'Tragic Valley'}\n")
                
                f.write("\n" + "=" * 60 + "\n")
            
            print(f"Created summary in: {summary_file}")
            print(f"\nTimestamped results directory created: {timestamp_dir}")


def main():
    # Process demo results
    find_and_process_experiments("demo_results")
    
    # Also check results directory
    if Path("results").exists():
        find_and_process_experiments("results")


if __name__ == "__main__":
    main()