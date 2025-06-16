#!/usr/bin/env python3
"""
Run All Experiments - Convenience script to run the complete analysis
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        return False
    return True

def main():
    """Run all experiments in sequence."""
    print("N-Person Prisoner's Dilemma - Complete Analysis")
    print("This will run all experiments and generate results.")
    
    # Check if results already exist
    if os.path.exists("results") or os.path.exists("qlearning_results"):
        response = input("\nResults directories already exist. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
    
    # List of experiments to run
    experiments = [
        ("python static_figure_generator.py", 
         "Static Strategy Experiments (TFT, AllC, AllD)"),
        
        ("python run_qlearning_demo.py", 
         "Basic Q-Learning Experiments"),
        
        ("python enhanced_qlearning_demo_generator.py", 
         "Enhanced Q-Learning Experiments"),
        
        ("python qlearning_comparative_analysis.py", 
         "Q-Learning Comparative Analysis")
    ]
    
    # Run each experiment
    for cmd, description in experiments:
        if not run_command(cmd, description):
            print("\nExperiment failed. Stopping.")
            return
    
    print("\n" + "="*60)
    print("All experiments completed successfully!")
    print("="*60)
    print("\nResults have been saved to:")
    print("  - results/                    (Static strategy results)")
    print("  - qlearning_results/          (Basic Q-learning results)")
    print("  - enhanced_qlearning_results/ (Enhanced Q-learning results)")
    print("  - comparative_analysis/       (Comparison between Q-learning variants)")
    print("\nCheck the folders for CSV data and PNG plots.")

if __name__ == "__main__":
    main()