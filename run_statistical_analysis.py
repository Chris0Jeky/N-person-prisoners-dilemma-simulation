import sys
import os
import json
import logging

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from npdl.analysis.analysis import compare_scenarios_stats
from npdl.core.logging_utils import setup_logging # For logging configuration

if __name__ == "__main__":
    # --- Configuration ---
    # List the scenarios (folder names in 'results/') you want to compare
    scenarios_to_compare = [
        "Baseline_QL_BasicState",      # Original Baseline
        "Baseline_QL_PropDiscr",       # Intermediate Baseline
        "HysQ_Opt_PropDiscr_vs_TFT",   # Best optimized Hysteretic-Q run
        "Wolf_Opt_PropDiscr_vs_TFT"    # Best optimized Wolf-PHC run
    ]

    results_directory = "results"  # Where the simulation outputs are stored
    output_directory = "analysis_results"  # Where to save comparison output
    metric_to_compare = 'final_cooperation_rate'  # Or 'average_final_score'
    significance_level = 0.05

    # --- Setup ---
    os.makedirs(output_directory, exist_ok=True)
    log_file = os.path.join(output_directory, "statistical_comparison.log")
    setup_logging(log_file=log_file, level=logging.INFO, console=True)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting statistical comparison for metric: {metric_to_compare}")
    logger.info(f"Scenarios: {', '.join(scenarios_to_compare)}")
    logger.info(f"Results directory: {results_directory}")
    logger.info(f"Significance level (alpha): {significance_level}")

    # --- Run Comparison ---
    comparison_results = compare_scenarios_stats(
        scenario_names=scenarios_to_compare,
        metric=metric_to_compare,
        results_dir=results_directory,
        alpha=significance_level
    )