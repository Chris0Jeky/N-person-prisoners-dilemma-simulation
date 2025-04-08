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

    # --- Output Results ---
    if comparison_results and "error" not in comparison_results:
        logger.info("\n--- Statistical Comparison Summary ---")

        logger.info(f"Metric Compared: {comparison_results.get('metric')}")
        logger.info(f"Scenarios Included: {comparison_results.get('scenarios_compared')}")

        anova_p = comparison_results.get('anova_p_value', None)
        if anova_p is not None:
            logger.info(f"\nANOVA Result:")
            logger.info(f"  F-value: {comparison_results.get('anova_f_value', 'N/A'):.4f}")
            logger.info(f"  p-value: {anova_p:.4g}")
            if anova_p < significance_level:
                logger.info("  Result: Significant difference found between scenarios (p < alpha).")
            else:
                logger.info("  Result: No significant overall difference found between scenarios (p >= alpha).")

        pairwise = comparison_results.get('pairwise_ttests', {})
        if pairwise:
            logger.info("\nPairwise t-test Results (Uncorrected):")
            for pair, res in pairwise.items():
                if "error" in res:
                    logger.info(f"  {pair}: Error - {res['error']}")
                else:
                    p_val = res.get('p_value', 1.0)
                    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < significance_level else "ns"
                    logger.info(f"  {pair}: t={res.get('t_value', 'N/A'):.3f}, p={p_val:.4g} ({significance})")
        logger.info("--------------------------------------")
