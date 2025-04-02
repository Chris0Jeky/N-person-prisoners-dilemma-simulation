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