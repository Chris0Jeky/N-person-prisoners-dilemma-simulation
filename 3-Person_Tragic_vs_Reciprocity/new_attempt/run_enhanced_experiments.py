#!/usr/bin/env python3
"""
Run the enhanced experiment runner with comprehensive logging and advanced visualizations.
This script runs all experiments and generates sophisticated comparative analyses.
"""

import enhanced_experiment_runner

if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED N-PERSON ITERATED PRISONER'S DILEMMA EXPERIMENTS")
    print("With Comprehensive Logging and Advanced Visualizations")
    print("=" * 80)
    print()
    
    # Run all experiments with detailed logging
    enhanced_experiment_runner.run_all_experiments_and_log()
    
    # Generate all visualizations
    if enhanced_experiment_runner.ALL_EXPERIMENT_RESULTS:
        print("\n" + "=" * 80)
        print("GENERATING ADVANCED VISUALIZATIONS")
        print("=" * 80)
        enhanced_experiment_runner.visualize_all_results(enhanced_experiment_runner.ALL_EXPERIMENT_RESULTS)
        
        print("\n" + "=" * 80)
        print("VISUALIZATION FILES CREATED:")
        print("- evolution_of_cooperation_comparison.png")
        print("- tft_performance_analysis.png") 
        print("- cooperation_dynamics_heatmap.png")
        print("- strategy_emergence_patterns.png")
        print("- summary_statistics_table.png")
        print("=" * 80)
    else:
        print("\nNo experiment results to visualize. Something went wrong!")
