#!/usr/bin/env python3
"""
Master script to run all experiments in the code_for_website directory.
This script runs each standalone experiment in sequence.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# List of all runnable scripts in the order they should be executed
SCRIPTS_TO_RUN = [
    # Static figure generators (basic experiments)
    ("static_figure_generator/static_figure_generator.py", 
     "Static Figure Generator - Full Version"),
    
    ("static_figure_generator_2tfte_allc_alld/static_figure_generator_2tfte_allc_alld.py", 
     "Static Figure Generator - 2 TFT-E Only"),
    
    # Q-learning experiments
    ("qlearning_demo/run_qlearning_demo.py", 
     "Q-Learning Demo"),
    
    # Main runs and analysis
    ("main_runs/save_config.py", 
     "Save Configuration"),
    
    ("main_runs/final_demo_full.py", 
     "Final Demo - Full Run"),
    
    # Specialized analyses
    ("cooperation_focussed/cooperation_measurement.py", 
     "Cooperation Measurement Analysis"),
    
    ("df_sensitivity/df_sensitivity_analysis.py", 
     "Discount Factor Sensitivity Analysis"),
]

def print_header(message):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f" {message} ")
    print("="*80 + "\n")

def print_separator():
    """Print a separator line."""
    print("\n" + "-"*80 + "\n")

def run_script(script_path, description):
    """Run a single script and handle errors."""
    print_header(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    full_path = os.path.join(os.path.dirname(__file__), script_path)
    
    if not os.path.exists(full_path):
        print(f"ERROR: Script not found: {full_path}")
        return False
    
    # Change to the script's directory to ensure relative imports work
    script_dir = os.path.dirname(full_path)
    original_dir = os.getcwd()
    
    try:
        os.chdir(script_dir)
        
        # Run the script
        start_time = time.time()
        result = subprocess.run([sys.executable, os.path.basename(full_path)], 
                              capture_output=True, text=True)
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ SUCCESS - Completed in {elapsed_time:.1f} seconds")
            if result.stdout:
                print("\nOutput preview (last 10 lines):")
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:
                    print(f"  {line}")
        else:
            print(f"✗ FAILED - Error after {elapsed_time:.1f} seconds")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print("\nError output:")
                print(result.stderr)
            if result.stdout:
                print("\nStandard output:")
                print(result.stdout)
            return False
            
    except Exception as e:
        print(f"✗ EXCEPTION: {type(e).__name__}: {e}")
        return False
    finally:
        # Always change back to original directory
        os.chdir(original_dir)
    
    return True

def main():
    """Run all experiments in sequence."""
    print_header("MASTER EXPERIMENT RUNNER")
    print("This script will run all experiments in the code_for_website directory.")
    print(f"Total number of experiments: {len(SCRIPTS_TO_RUN)}")
    print("\nNote: This may take a considerable amount of time to complete.")
    print("Each experiment will generate its own output files in its respective directory.")
    
    # Ask for confirmation
    response = input("\nDo you want to continue? (y/n): ").lower().strip()
    if response != 'y':
        print("Aborted by user.")
        return
    
    # Track results
    results = []
    start_time = time.time()
    
    # Run each script
    for i, (script_path, description) in enumerate(SCRIPTS_TO_RUN, 1):
        print_separator()
        print(f"Progress: {i}/{len(SCRIPTS_TO_RUN)}")
        
        success = run_script(script_path, description)
        results.append((script_path, description, success))
        
        # Small delay between scripts
        if i < len(SCRIPTS_TO_RUN):
            time.sleep(2)
    
    # Print summary
    total_time = time.time() - start_time
    print_header("SUMMARY")
    print(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\nResults:")
    
    successful = 0
    failed = 0
    
    for script_path, description, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status} - {description}")
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\nTotal: {successful} successful, {failed} failed")
    
    # List output directories
    print("\nOutput locations:")
    print("  - static_figure_generator/results/")
    print("  - static_figure_generator_2tfte_allc_alld/results_2tfte_only/")
    print("  - qlearning_demo/qlearning_results/")
    print("  - main_runs/results/")
    print("  - cooperation_focussed/results/")
    print("  - df_sensitivity/results/")
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main()