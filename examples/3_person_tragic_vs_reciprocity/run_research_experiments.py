#!/usr/bin/env python3
"""
Run comprehensive research experiments for TFT-centric and QLearning-centric analysis.
This script runs directly from the main directory.
"""

import sys
import os

# Add npd_simulator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'npd_simulator'))

# Now we can import
from experiments.research_experiments import ResearchExperiments
import logging

def main():
    """Run all research experiments."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Starting comprehensive research experiments...")
    print("=" * 60)
    
    # Initialize experiments
    experiments = ResearchExperiments(
        num_runs=20,      # 20 runs for statistical significance
        num_rounds=200,   # 200 rounds per experiment
        output_dir="results"
    )
    
    print("\nExperiment Configuration:")
    print(f"- Number of runs per experiment: {experiments.num_runs}")
    print(f"- Number of rounds per run: {experiments.num_rounds}")
    print(f"- Output directory: {experiments.output_dir}")
    
    print("\nPhase 1: TFT-Centric Experiments")
    print("-" * 40)
    print("Running experiments to demonstrate tragic valley vs reciprocity hill...")
    
    try:
        # Run TFT experiments
        tft_results = experiments.run_tft_centric_experiments()
        print("✓ TFT-centric experiments completed")
        
        print("\nPhase 2: QLearning-Centric Experiments")
        print("-" * 40)
        print("Running QLearning experiments with various agent combinations...")
        
        # Run QLearning experiments
        ql_results = experiments.run_qlearning_centric_experiments()
        print("✓ QLearning-centric experiments completed")
        
        print("\nPhase 3: Analysis and Visualization")
        print("-" * 40)
        
        # Generate analysis
        analysis = experiments.generate_comparative_analysis(tft_results, ql_results)
        print("✓ Comparative analysis generated")
        
        # Generate visualizations
        experiments._generate_visualizations(tft_results, ql_results)
        print("✓ Visualizations created")
        
        print("\n" + "=" * 60)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nResults saved to 'results/' directory:")
        print("- TFT cooperation CSVs (pairwise and neighbourhood)")
        print("- QLearning cooperation CSVs (2QL and 1QL variants)")
        print("- Summary comparison CSVs")
        print("- Comparative analysis JSON")
        print("- Visualization figures")
        
        # Print key insights
        if "comparative_insights" in analysis:
            insights = analysis["comparative_insights"]
            print("\nKey Findings:")
            if insights.get("tragic_valley_observed"):
                print("✓ Tragic valley effect observed")
            if insights.get("reciprocity_hill_observed"):
                print("✓ Reciprocity hill effect observed")
            
            for finding in insights.get("key_findings", [])[:3]:
                print(f"- {finding}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())