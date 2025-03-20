"""
Data loading utilities for visualization dashboard.

This module contains functions for loading simulation results
and preprocessing data for visualization.
"""

import os
import pandas as pd
import json
import glob
from typing import Dict, List, Tuple, Optional


def get_available_scenarios(results_dir: str = "results") -> List[str]:
    """Get list of all available scenario names from results directory."""
    # Find all agent CSV files
    agent_files = glob.glob(os.path.join(results_dir, "*_agents.csv"))
    
    # Extract scenario names from filenames
    scenarios = []
    for file_path in agent_files:
        basename = os.path.basename(file_path)
        if "_agents.csv" in basename:
            scenario = basename.replace("experiment_results_", "").replace("_agents.csv", "")
            scenarios.append(scenario)
    
    return sorted(scenarios)


def load_scenario_results(scenario_name: str, results_dir: str = "results", 
                         base_filename: str = "experiment_results") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load experiment results for a given scenario."""
    agents_file = os.path.join(results_dir, f"{base_filename}_{scenario_name}_agents.csv")
    rounds_file = os.path.join(results_dir, f"{base_filename}_{scenario_name}_rounds.csv")
    
    if not os.path.exists(agents_file) or not os.path.exists(rounds_file):
        raise FileNotFoundError(f"Results files not found for scenario: {scenario_name}")
    
    agents_df = pd.read_csv(agents_file)
    rounds_df = pd.read_csv(rounds_file)
    
    return agents_df, rounds_df


def get_cooperation_rates(rounds_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cooperation rates by round."""
    # Add cooperation column (1 for cooperate, 0 for defect)
    rounds_df['cooperation'] = (rounds_df['move'] == 'cooperate').astype(int)
    
    # Group by round and calculate cooperation rate
    coop_rates = rounds_df.groupby('round')['cooperation'].mean().reset_index()
    coop_rates.rename(columns={'cooperation': 'cooperation_rate'}, inplace=True)
    
    return coop_rates


def get_strategy_cooperation_rates(rounds_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cooperation rates by round and strategy."""
    # Add cooperation column (1 for cooperate, 0 for defect)
    rounds_df['cooperation'] = (rounds_df['move'] == 'cooperate').astype(int)
    
    # Group by round and strategy, calculate cooperation rate
    coop_rates = rounds_df.groupby(['round', 'strategy'])['cooperation'].mean().reset_index()
    coop_rates.rename(columns={'cooperation': 'cooperation_rate'}, inplace=True)
    
    return coop_rates
