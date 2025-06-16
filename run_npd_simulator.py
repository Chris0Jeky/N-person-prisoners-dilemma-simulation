#!/usr/bin/env python3
"""
Helper script to run NPD simulator modules from the parent directory.
This script sets up the Python path correctly to avoid import errors.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Now we can import and run various NPD simulator components
def run_demo():
    """Run the NPD simulator demo."""
    from npd_simulator.demo import demo_single_experiment, demo_qlearning_comparison
    print("Running NPD Simulator Demo...")
    demo_single_experiment()
    demo_qlearning_comparison()

def run_main(args):
    """Run the main NPD simulator with arguments."""
    from npd_simulator.main import main
    # Pass command line arguments to main
    sys.argv = ['main.py'] + args
    main()

def run_research_experiments():
    """Run research experiments."""
    from npd_simulator.experiments.research_experiments import ResearchExperiments
    print("Running Research Experiments...")
    research = ResearchExperiments(num_runs=5, num_rounds=100)
    research.run_tft_centric_experiments()

def run_test_static_style():
    """Run static style tests."""
    from npd_simulator.test_static_style import test_with_demo_results
    print("Running Static Style Tests...")
    test_with_demo_results()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run NPD Simulator components")
    parser.add_argument("command", choices=["demo", "main", "research", "test-static"],
                        help="Which component to run")
    parser.add_argument("args", nargs="*", help="Additional arguments for the command")
    
    args = parser.parse_args()
    
    if args.command == "demo":
        run_demo()
    elif args.command == "main":
        run_main(args.args)
    elif args.command == "research":
        run_research_experiments()
    elif args.command == "test-static":
        run_test_static_style()