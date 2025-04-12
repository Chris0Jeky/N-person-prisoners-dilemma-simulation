#!/usr/bin/env python3
"""
Run a complete scenario sweep and analysis workflow.

This script:
1. Generates and evaluates random scenarios
2. Selects the most interesting ones
3. Runs and saves detailed results for the selected scenarios
4. Generates visualization and analysis of the results
"""
import os
import argparse
import time
from run_scenario_generator import run_scenario_generation
from analysis.sweep_visualizer import create_scenario_comparison_report


def run_sweep_and_analysis(num_generate=30, 
                          eval_runs=3, 
                          save_runs=10, 
                          top_n=5,
                          results_dir="results/generated_scenarios",
                          analysis_dir="analysis_results",
                          log_level="INFO"):
    """Run the complete workflow of scenario generation, evaluation, and analysis."""
    print(f"=== Starting Scenario Sweep Analysis ===")
    print(f"Generating {num_generate} scenarios, selecting top {top_n}")
    print(f"Results will be saved in: {results_dir}")
    print(f"Analysis will be saved in: {analysis_dir}")
    
    start_time = time.time()
    
    # Step 1: Generate and evaluate scenarios
    run_scenario_generation(
        num_scenarios_to_generate=num_generate,
        num_eval_runs=eval_runs,
        num_save_runs=save_runs,
        top_n_to_save=top_n,
        results_dir=results_dir,
        log_level_str=log_level
    )
    
    # Step 2: Analyze and visualize results
    metadata_path = os.path.join(results_dir, "generated_scenarios_metadata.json")
    if os.path.exists(metadata_path):
        create_scenario_comparison_report(metadata_path, analysis_dir)
    else:
        print(f"Warning: Metadata file not found at {metadata_path}")
        print("Analysis step skipped.")
    
    end_time = time.time()
    print(f"=== Scenario Sweep Analysis Completed ===")
    print(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scenario sweep and analysis")
    parser.add_argument("--num_generate", type=int, default=30,
                       help="Number of random scenarios to generate")
    parser.add_argument("--eval_runs", type=int, default=3,
                       help="Number of evaluation runs per scenario")
    parser.add_argument("--save_runs", type=int, default=10,
                       help="Number of full runs for selected scenarios")
    parser.add_argument("--top_n", type=int, default=5,
                       help="Number of top scenarios to save and analyze")
    parser.add_argument("--results_dir", type=str, default="results/generated_scenarios",
                       help="Directory to save scenario results")
    parser.add_argument("--analysis_dir", type=str, default="analysis_results",
                       help="Directory to save analysis results")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    run_sweep_and_analysis(
        num_generate=args.num_generate,
        eval_runs=args.eval_runs,
        save_runs=args.save_runs,
        top_n=args.top_n,
        results_dir=args.results_dir,
        analysis_dir=args.analysis_dir,
        log_level=args.log_level
    )
