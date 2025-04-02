# run_parameter_sweep.py
import argparse
import json
import os
import itertools
import random
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any

# Adjust imports based on your final structure if main.py was split
from main import setup_experiment # We need this to setup agents/env per combo
from npdl.core.logging_utils import setup_logging

# --- Helper function to extract performance ---
def calculate_performance_metrics(env, round_results, target_strategy):
    """Calculates key performance metrics from a single simulation run."""
    metrics = {}
    # Final Cooperation Rate (Overall)
    if round_results:
        last_round = round_results[-1]
        coop_count = sum(1 for move in last_round['moves'].values() if move == "cooperate")
        metrics['final_coop_rate_overall'] = coop_count / len(last_round['moves'])
    else:
        metrics['final_coop_rate_overall'] = 0.0

        # Average Final Score of Target Strategy Agents
        target_agents = [a for a in env.agents if a.strategy_type == target_strategy]
        if target_agents:
            metrics['avg_final_score_target'] = sum(a.score for a in target_agents) / len(target_agents)
        else:
            metrics['avg_final_score_target'] = 0.0  # Or NaN if preferred

        # Can add more metrics here if needed (e.g., final score of other strategies)
        return metrics

# --- Main Sweep Function ---
def run_sweep(config):
    """Runs the parameter sweep based on the configuration."""

    base_scenario = config['base_scenario']
    param_grid = config['parameter_grid']
    target_strategy = config['target_strategy']
    num_runs = config['num_runs_per_combo']
    output_file = config['output_file']
    log_level_str = config.get('log_level', 'INFO') # Allow setting log level

    # Setup basic logging for the sweep process
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    sweep_logger = setup_logging(level=log_level, console=True,
                                 log_file=f"sweep_{target_strategy}.log")  # Log sweep progress

    sweep_logger.info(f"Starting parameter sweep for strategy: {target_strategy}")
    sweep_logger.info(f"Base scenario: {base_scenario['scenario_name']}")
    sweep_logger.info(f"Parameter grid: {param_grid}")
    sweep_logger.info(f"Runs per combination: {num_runs}")

    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    sweep_logger.info(f"Total combinations to test: {len(combinations)}")

    results_list = []
    total_simulations = len(combinations) * num_runs
    completed_simulations = 0
    start_sweep_time = time.time()

    for i, combo_values in enumerate(combinations):
        combo_params = dict(zip(param_names, combo_values))
        sweep_logger.info(f"Testing Combination {i + 1}/{len(combinations)}: {combo_params}")

        # Create the specific scenario for this combination
        current_scenario = base_scenario.copy()
        # Update parameters that are being swept
        current_scenario.update(combo_params)

        # Ensure target strategy is present (might override base if needed)
        if target_strategy not in current_scenario['agent_strategies']:
            # If target wasn't in base, add it (or handle this error)
            # This assumes the base scenario has other agents defined
            current_scenario['agent_strategies'][target_strategy] = 1  # Add one agent? Needs careful thought
            current_scenario['num_agents'] = sum(current_scenario['agent_strategies'].values())
            sweep_logger.warning(
                f"Target strategy '{target_strategy}' not in base scenario, adding 1 agent. Adjust num_agents.")

        # Store results for this combination across runs
        combo_run_metrics = []

        for run_number in range(num_runs):
            # Set seed for reproducibility *within this specific run*
            seed = (i * num_runs) + run_number  # Ensure unique seed per sim run
            random.seed(seed)
            np.random.seed(seed)

            # Setup FRESH agents and environment for EACH run
            # Use a temporary logger that doesn't clutter the main log file excessively during runs
            # Or pass the sweep_logger but rely on its level setting
            try:
                env, _ = setup_experiment(current_scenario, sweep_logger)

                # Run simulation *without* saving individual run artifacts
                # Set logging_interval high to minimize log spam during sweep
                round_results = env.run_simulation(
                    current_scenario["num_rounds"],
                    logging_interval=current_scenario["num_rounds"] + 1,  # Effectively disable round logging
                    use_global_bonus=current_scenario.get("use_global_bonus", False),
                    # Add other relevant params if they are in base_scenario
                    rewiring_interval=current_scenario.get("rewiring_interval", 0),
                    rewiring_prob=current_scenario.get("rewiring_prob", 0.0)
                )

                # Calculate performance metrics for this run
                metrics = calculate_performance_metrics(env, round_results, target_strategy)
                combo_run_metrics.append(metrics)
                completed_simulations += 1
                progress = (completed_simulations / total_simulations) * 100
                sweep_logger.debug(f"  Run {run_number + 1}/{num_runs} complete. Progress: {progress:.1f}%")

            except Exception as e:
                sweep_logger.error(f"  ERROR during run {run_number + 1} for combo {combo_params}: {e}", exc_info=False)
                # Append placeholder or skip run? Let's skip for average calculation
                continue  # Skip to next run

        # Aggregate metrics for this combination
        if combo_run_metrics:  # Only if at least one run succeeded
            avg_metrics = {}
            metric_keys = combo_run_metrics[0].keys()
            for key in metric_keys:
                values = [m[key] for m in combo_run_metrics]
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)
        else:
            avg_metrics = {}  # No successful runs for this combo

        # Store results
        results_list.append({**combo_params, **avg_metrics})

    # Create DataFrame and sort by performance
    results_df = pd.DataFrame(results_list)
    # Sort by desired metric, e.g., highest avg final cooperation rate
    primary_metric = f'avg_final_coop_rate_overall'
    if primary_metric in results_df.columns:
        results_df = results_df.sort_values(by=primary_metric, ascending=False)
    else:
        sweep_logger.warning(f"Primary metric '{primary_metric}' not found for sorting.")

    # Save results to CSV
    try:
        results_df.to_csv(output_file, index=False)
        sweep_logger.info(f"Sweep results saved to {output_file}")
    except Exception as e:
        sweep_logger.error(f"Failed to save sweep results to {output_file}: {e}")

    # Print top results
    sweep_logger.info(f"\n--- Top Performing Combinations for {target_strategy} ---")
    sweep_logger.info(f"(Sorted by {primary_metric})\n")
    # Use pandas toString() for better formatting in log/console
    top_n = 10
    sweep_logger.info(results_df.head(top_n).to_string())
    sweep_logger.info("----------------------------------------------------\n")

    end_sweep_time = time.time()
    sweep_logger.info(f"Parameter sweep completed in {end_sweep_time - start_sweep_time:.2f} seconds.")

# --- Argument Parsing and Configuration ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameter sweep for N-IPD agent strategies.")
    parser.add_argument('--strategy', type=str, required=True,
                        help='Target agent strategy type to optimize (e.g., lra_q, hysteretic_q).')
    parser.add_argument('--config', type=str, default='sweep_config.json',
                        help='Path to the JSON configuration file for the sweep.')
    args = parser.parse_args()

    # Load sweep configuration from JSON
    try:
        with open(args.config, 'r') as f:
            sweep_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Sweep configuration file '{args.config}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON in '{args.config}'.")
        exit(1)
























