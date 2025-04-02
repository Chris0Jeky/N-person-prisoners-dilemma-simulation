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
from main import setup_experiment
from npdl.core.logging_utils import setup_logging

# --- Helper function to extract performance ---
def calculate_performance_metrics(env, round_results, target_strategy):
    """Calculates key performance metrics from a single simulation run."""
    metrics = {}
    if not round_results: # Handle case where simulation might have failed early
        metrics['final_coop_rate_overall'] = np.nan
        metrics['avg_final_score_target'] = np.nan
        return metrics

    # Final Cooperation Rate (Overall)
    last_round = round_results[-1]
    if last_round['moves']:
        coop_count = sum(1 for move in last_round['moves'].values() if move == "cooperate")
        metrics['final_coop_rate_overall'] = coop_count / len(last_round['moves'])
    else:
        metrics['final_coop_rate_overall'] = np.nan # Should not happen if runs succeed

    # Average Final Score of Target Strategy Agents
    target_agents = [a for a in env.agents if a.strategy_type == target_strategy]
    if target_agents:
        metrics['avg_final_score_target'] = sum(a.score for a in target_agents) / len(target_agents)
    else:
        metrics['avg_final_score_target'] = np.nan # Target strategy wasn't found (config error)

    # Add more metrics here if needed
    return metrics

# --- Function to run sweep for a SINGLE strategy ---
def run_single_strategy_sweep(config, target_strategy, strategy_config, base_scenario_template, global_settings):
    """Runs the parameter sweep for one specified strategy."""

    parameter_grid = strategy_config['parameter_grid']
    target_agent_count = strategy_config['target_agent_count']
    num_runs = global_settings['num_runs_per_combo']
    output_base_dir = global_settings['output_base_dir']
    log_level_str = global_settings.get('log_level', 'INFO')

    # --- Output Directory and Logging ---
    os.makedirs(output_base_dir, exist_ok=True)
    output_csv_file = os.path.join(output_base_dir, f"sweep_results_{target_strategy}.csv")
    log_file = os.path.join(output_base_dir, f"sweep_log_{target_strategy}.log")

    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    # Create a unique logger for this strategy's sweep
    sweep_logger = setup_logging(level=log_level, console=False, log_file=log_file) # Log to file only

    print(f"\n===== Starting Sweep for: {target_strategy} =====") # Console feedback
    sweep_logger.info(f"Starting parameter sweep for strategy: {target_strategy}")
    sweep_logger.info(f"Parameter grid: {parameter_grid}")
    sweep_logger.info(f"Runs per combination: {num_runs}")
    sweep_logger.info(f"Results will be saved to: {output_csv_file}")
    sweep_logger.info(f"Detailed log in: {log_file}")

    # Generate parameter combinations
    param_names = list(parameter_grid.keys())
    param_values = list(parameter_grid.values())
    combinations = list(itertools.product(*param_values))

    sweep_logger.info(f"Total combinations to test: {len(combinations)}")

    results_list = []
    total_simulations = len(combinations) * num_runs
    completed_simulations = 0
    start_sweep_time = time.time()

    for i, combo_values in enumerate(combinations):
        combo_params = dict(zip(param_names, combo_values))
        sweep_logger.info(f"Testing Combination {i+1}/{len(combinations)}: {combo_params}")
        print(f"  Strategy: {target_strategy} | Combo {i+1}/{len(combinations)}: {combo_params}") # Console feedback

        # --- Create the specific scenario for this combination ---
        current_scenario = base_scenario_template.copy() # Start with base
        current_scenario['scenario_name'] = f"{base_scenario_template['scenario_name_prefix']}_{target_strategy}_Combo{i+1}"

        # Apply strategy-specific overrides from config (e.g., state_type)
        for key, value in strategy_config.items():
             if key.endswith("_override"):
                 original_key = key.replace("_override", "")
                 current_scenario[original_key] = value
                 sweep_logger.debug(f"    Overriding '{original_key}' with '{value}' for {target_strategy}")

        # Apply the current parameter combination being swept
        current_scenario.update(combo_params)

        # Set agent strategies (fixed opponents + target strategy with count)
        current_scenario['agent_strategies'] = base_scenario_template.get('fixed_opponents', {}).copy()
        current_scenario['agent_strategies'][target_strategy] = target_agent_count
        current_scenario['num_agents'] = sum(current_scenario['agent_strategies'].values())
        # --- End Scenario Creation ---

        combo_run_metrics = []
        for run_number in range(num_runs):
            seed = (i * num_runs) + run_number # Unique seed
            random.seed(seed)
            np.random.seed(seed)

            # Create a temporary silent logger for setup_experiment to avoid spam
            silent_logger = logging.getLogger('silent_setup')
            silent_logger.setLevel(logging.ERROR) # Or WARNING

            try:
                env, _ = setup_experiment(current_scenario, silent_logger)

                # Run simulation without excessive logging/saving
                round_results = env.run_simulation(
                    current_scenario["num_rounds"],
                    logging_interval=current_scenario["num_rounds"] + 1,
                    use_global_bonus=current_scenario.get("use_global_bonus", False),
                    rewiring_interval=current_scenario.get("rewiring_interval", 0),
                    rewiring_prob=current_scenario.get("rewiring_prob", 0.0)
                )

                metrics = calculate_performance_metrics(env, round_results, target_strategy)
                combo_run_metrics.append(metrics)
                completed_simulations += 1

            except Exception as e:
                 sweep_logger.error(f"  ERROR during run {run_number+1} for combo {combo_params}: {e}", exc_info=False)
                 combo_run_metrics.append({}) # Append empty dict or nan dict on error? Let's append empty.

        # Aggregate metrics for this combination
        avg_metrics = {}
        valid_run_metrics = [m for m in combo_run_metrics if isinstance(m, dict) and m] # Filter empty dicts too

        if valid_run_metrics:
            all_keys = set()
            for m in valid_run_metrics:
                all_keys.update(m.keys())
            metric_keys = list(all_keys)

            for key in metric_keys:
                values = [m[key] for m in valid_run_metrics if key in m and not np.isnan(m[key])] # Exclude NaNs from calcs
                if values:
                    avg_metrics[f'avg_{key}'] = np.mean(values)
                    avg_metrics[f'std_{key}'] = np.std(values)
                else:
                    avg_metrics[f'avg_{key}'] = np.nan
                    avg_metrics[f'std_{key}'] = np.nan

        results_list.append({**combo_params, **avg_metrics})
        progress = ( (i+1) / len(combinations) ) * 100
        print(f"  Strategy: {target_strategy} | Combo {i+1}/{len(combinations)} complete. Progress: {progress:.1f}%") # Console feedback


    # --- Save and Summarize Results for this Strategy ---
    if not results_list:
        sweep_logger.warning(f"No results generated for strategy {target_strategy}.")
        print(f"===== No results generated for: {target_strategy} =====")
        return

    results_df = pd.DataFrame(results_list)
    primary_metric = 'avg_final_coop_rate_overall'
    if primary_metric in results_df.columns:
        results_df = results_df.sort_values(by=primary_metric, ascending=False)
    else:
        sweep_logger.warning(f"Primary metric '{primary_metric}' not found for sorting {target_strategy}.")

    try:
        results_df.to_csv(output_csv_file, index=False)
        sweep_logger.info(f"Sweep results for {target_strategy} saved to {output_csv_file}")
        print(f"Results for {target_strategy} saved to {output_csv_file}")
    except Exception as e:
        sweep_logger.error(f"Failed to save sweep results for {target_strategy} to {output_csv_file}: {e}")
        print(f"Error saving results for {target_strategy}: {e}")


    sweep_logger.info(f"\n--- Top Performing Combinations for {target_strategy} ---")
    sweep_logger.info(f"(Sorted by {primary_metric})\n")
    top_n = 10
    sweep_logger.info(results_df.head(top_n).to_string())
    sweep_logger.info("----------------------------------------------------\n")

    end_sweep_time = time.time()
    sweep_logger.info(f"Sweep for {target_strategy} completed in {end_sweep_time - start_sweep_time:.2f} seconds.")
    print(f"===== Sweep for: {target_strategy} Complete =====")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameter sweeps for N-IPD agent strategies based on a config file.")
    parser.add_argument('--config', type=str, default='multi_sweep_config.json',
                        help='Path to the JSON configuration file defining multiple sweeps.')
    args = parser.parse_args()

    # Load sweep configuration from JSON
    try:
        with open(args.config, 'r') as f:
            sweep_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Sweep configuration file '{args.config}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse JSON in '{args.config}': {e}")
        exit(1)

    # --- Validate Config Structure ---
    if "global_settings" not in sweep_config or \
       "base_scenario" not in sweep_config or \
       "strategy_sweeps" not in sweep_config:
        print("Error: Configuration file missing required top-level keys: 'global_settings', 'base_scenario', 'strategy_sweeps'.")
        exit(1)

    global_settings = sweep_config['global_settings']
    base_scenario_template = sweep_config['base_scenario']
    strategy_sweeps = sweep_config['strategy_sweeps']

    print(f"Loaded sweep configuration from: {args.config}")
    print(f"Output directory: {global_settings.get('output_base_dir', 'parameter_sweep_results')}")
    print(f"Strategies to sweep: {list(strategy_sweeps.keys())}")

    # --- Run Sweep for Each Strategy Defined in Config ---
    overall_start_time = time.time()
    for strategy_name, strategy_details in strategy_sweeps.items():
        if "parameter_grid" not in strategy_details or "target_agent_count" not in strategy_details:
            print(f"Warning: Skipping strategy '{strategy_name}' due to missing 'parameter_grid' or 'target_agent_count' in config.")
            continue
        run_single_strategy_sweep(sweep_config, strategy_name, strategy_details, base_scenario_template, global_settings)

    overall_end_time = time.time()
    print(f"\nAll parameter sweeps finished in {overall_end_time - overall_start_time:.2f} seconds.")