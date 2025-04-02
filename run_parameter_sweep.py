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

























