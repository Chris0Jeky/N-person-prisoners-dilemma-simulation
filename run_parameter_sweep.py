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