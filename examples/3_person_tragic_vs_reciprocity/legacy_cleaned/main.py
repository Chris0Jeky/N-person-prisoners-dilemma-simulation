#!/usr/bin/env python3
"""
Main entry point for N-Person Prisoner's Dilemma experiments
"""

import argparse
import sys
from src import ExperimentRunner, run_qlearning_experiments

def main():
    parser = argparse.ArgumentParser(
        description="N-Person Prisoner's Dilemma Simulation Framework"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["default", "qlearning", "custom"],
        default="default",
        help="Experiment mode to run"
    )
    
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of simulation runs (default: 10)"
    )
    
    parser.add_argument(
        "--rounds",
        type=int,
        default=200,
        help="Number of rounds per simulation (default: 200)"
    )
    
    parser.add_argument(
        "--exploration-rates",
        nargs="+",
        type=float,
        default=[0.0, 0.1],
        help="Exploration rates to test (default: 0.0 0.1)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting N-Person Prisoner's Dilemma Experiments")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Rounds per simulation: {args.rounds}")
    print("-" * 50)
    
    if args.mode == "qlearning":
        # Run Q-learning specific experiments
        run_qlearning_experiments(
            output_dir=args.output_dir,
            num_runs=args.num_runs,
            training_rounds=1000
        )
    
    elif args.mode == "custom":
        # Allow for custom configurations
        print("Custom mode: Edit main.py to define your own experiments")
        # Add your custom experiment code here
        
    else:  # default mode
        # Run standard experiments
        runner = ExperimentRunner(base_output_dir=args.output_dir)
        runner.exploration_rates = args.exploration_rates
        runner.num_runs = args.num_runs
        runner.nperson_rounds = args.rounds
        runner.pairwise_rounds = min(args.rounds, 100)
        
        runner.run_all_experiments()
    
    print("\nAll experiments completed successfully!")
    print(f"Results saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()