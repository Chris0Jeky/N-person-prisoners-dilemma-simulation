#!/usr/bin/env python3
"""
Quick test version of research experiments with reduced parameters.
"""

import sys
import os

# Add npd_simulator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'npd_simulator'))

from experiments.research_experiments import ResearchExperiments
import logging

def main():
    """Run quick test experiments."""
    logging.basicConfig(level=logging.INFO)
    
    print("Running QUICK TEST version (reduced parameters)...")
    
    # Initialize with reduced parameters
    experiments = ResearchExperiments(
        num_runs=3,       # Only 3 runs
        num_rounds=20,    # Only 20 rounds
        output_dir="test_results"
    )
    
    print(f"Test configuration: {experiments.num_runs} runs, {experiments.num_rounds} rounds")
    
    # Test just one TFT experiment
    print("\nTesting TFT experiment...")
    test_config = {
        "3_TFT": experiments.tft_configs["3_TFT"]
    }
    experiments.tft_configs = test_config
    tft_results = experiments.run_tft_centric_experiments()
    
    # Test just one QL experiment
    print("\nTesting QL experiment...")
    experiments.other_agent_types = ["AllD"]  # Just one type
    ql_results = experiments.run_qlearning_centric_experiments()
    
    print("\nGenerating analysis...")
    analysis = experiments.generate_comparative_analysis(tft_results, ql_results)
    
    print("\nQuick test completed! Check 'test_results/' directory.")
    
    # Print what was generated
    import os
    if os.path.exists("test_results"):
        files = os.listdir("test_results")
        print(f"\nGenerated {len(files)} files:")
        for f in sorted(files)[:10]:
            print(f"  - {f}")

if __name__ == "__main__":
    main()