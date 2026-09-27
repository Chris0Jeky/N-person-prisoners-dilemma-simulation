#!/usr/bin/env python3
"""
Run a complete scenario sweep and analysis workflow.

This script:
1. Generates and evaluates random scenarios
2. Selects the most interesting ones
3. Runs and saves detailed results for the selected scenarios
4. Generates visualization and analysis of the results
"""

import argparse
import os
import time

from run_scenario_generator import run_scenario_generation

from npdl.experiments import create_run

try:
    # Optional: absent from npdl.analysis.sweep_visualizer (which only ships
    # visualize_sweep_results) and needs seaborn; the analysis step is
    # skipped gracefully when unavailable, mirroring the missing-metadata path.
    from npdl.analysis.sweep_visualizer import create_scenario_comparison_report
except ImportError:
    create_scenario_comparison_report = None


def run_sweep_and_analysis(
    num_generate=30,
    eval_runs=3,
    save_runs=10,
    top_n=5,
    results_dir="results/generated_scenarios",
    analysis_dir="analysis_results",
    log_level="INFO",
    seed=0,
):
    """Run the complete workflow of scenario generation, evaluation, and analysis.

    Both stages are registered: the generation stage writes its own run dir
    (see :func:`run_scenario_generation`) and the analysis stage registers
    ``analysis_dir`` with its own seed/config-hash/manifest sidecars.
    """
    print(f"=== Starting Scenario Sweep Analysis ===")
    print(f"Generating {num_generate} scenarios, selecting top {top_n}")
    print(f"Results will be saved in: {results_dir}")
    print(f"Analysis will be saved in: {analysis_dir}")
    print(f"Run seed: {seed}")

    start_time = time.time()

    # Step 1: Generate and evaluate scenarios (registered run of its own)
    run_scenario_generation(
        num_scenarios_to_generate=num_generate,
        num_eval_runs=eval_runs,
        num_save_runs=save_runs,
        top_n_to_save=top_n,
        results_dir=results_dir,
        log_level_str=log_level,
        seed=seed,
    )

    # Step 2: Analyze and visualize results (registered as its own run dir)
    analysis_dir = os.path.abspath(analysis_dir)
    analysis_run = create_run(
        os.path.dirname(analysis_dir),
        "sweep_analysis",
        {
            "num_generate": num_generate,
            "eval_runs": eval_runs,
            "save_runs": save_runs,
            "top_n": top_n,
            "results_dir": results_dir,
            "analysis_dir": os.path.basename(analysis_dir),
            "log_level": log_level,
            "seed": seed,
        },
        seed,
        run_name=os.path.basename(analysis_dir),
    )
    metadata_path = os.path.join(
        os.path.abspath(results_dir), "generated_scenarios_metadata.json"
    )
    if os.path.exists(metadata_path) and create_scenario_comparison_report is not None:
        create_scenario_comparison_report(metadata_path, analysis_run.run_dir)
    else:
        if create_scenario_comparison_report is None:
            print(
                "Warning: scenario comparison helper unavailable; analysis step skipped."
            )
        else:
            print(f"Warning: Metadata file not found at {metadata_path}")
            print("Analysis step skipped.")
    manifest = analysis_run.finalize()
    print(f"Analysis manifest written with {len(manifest['artifacts'])} artifacts.")

    end_time = time.time()
    print(f"=== Scenario Sweep Analysis Completed ===")
    print(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scenario sweep and analysis")
    parser.add_argument(
        "--num_generate",
        type=int,
        default=30,
        help="Number of random scenarios to generate",
    )
    parser.add_argument(
        "--eval_runs",
        type=int,
        default=3,
        help="Number of evaluation runs per scenario",
    )
    parser.add_argument(
        "--save_runs",
        type=int,
        default=10,
        help="Number of full runs for selected scenarios",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of top scenarios to save and analyze",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/generated_scenarios",
        help="Directory to save scenario results",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default="analysis_results",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Run seed shared by both stages; re-running with the same seed reproduces manifests",
    )

    args = parser.parse_args()

    run_sweep_and_analysis(
        num_generate=args.num_generate,
        eval_runs=args.eval_runs,
        save_runs=args.save_runs,
        top_n=args.top_n,
        results_dir=args.results_dir,
        analysis_dir=args.analysis_dir,
        log_level=args.log_level,
        seed=args.seed,
    )
