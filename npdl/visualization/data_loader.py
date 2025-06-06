"""
Data loading utilities for visualization dashboard.

This module contains functions for loading simulation results
and preprocessing data for visualization.
"""

import os
import pandas as pd
import json
import glob
import logging
import networkx as nx
from typing import Dict, List, Tuple, Optional


def get_available_scenarios(results_dir: str = "results") -> List[str]:
    """Get list of all available scenario names from results directory.

    Args:
        results_dir: Directory containing the results

    Returns:
        List of scenario names found in the directory
    """
    # Check for scenario directories
    try:
        scenario_dirs = [
            d
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d)) and not d.startswith("_")
        ]
        return sorted(scenario_dirs)
    except FileNotFoundError:
        logging.warning(f"Results directory '{results_dir}' not found")
        return []


def load_scenario_results(
    scenario_name: str,
    results_dir: str = "results",
    base_filename: str = "experiment_results",
    run_number: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load experiment results for a given scenario.

    Args:
        scenario_name: Name of the scenario
        results_dir: Directory containing result files
        base_filename: Base filename prefix for result files
        run_number: Specific run to load (if None, aggregates all runs)

    Returns:
        Tuple of (agents_df, rounds_df)

    Raises:
        FileNotFoundError: If the requested scenario or run files cannot be found
    """
    scenario_dir = os.path.join(results_dir, scenario_name)

    if not os.path.isdir(scenario_dir):
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    # Multi-run structure
    all_agents = []
    all_rounds = []

    if run_number is not None:
        # Load specific run
        run_dir = os.path.join(scenario_dir, f"run_{run_number:02d}")
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        agents_file = os.path.join(run_dir, f"{base_filename}_agents.csv")
        rounds_file = os.path.join(run_dir, f"{base_filename}_rounds.csv")

        if not os.path.exists(agents_file) or not os.path.exists(rounds_file):
            raise FileNotFoundError(
                f"Results files not found for run {run_number} of scenario: {scenario_name}"
            )

        agents_df = pd.read_csv(agents_file)
        rounds_df = pd.read_csv(rounds_file)
        return agents_df, rounds_df
    else:
        # Aggregate all runs
        run_dirs = [d for d in os.listdir(scenario_dir) if d.startswith("run_")]
        if not run_dirs:
            raise FileNotFoundError(
                f"No run directories found in scenario: {scenario_name}"
            )

        for run_dir_name in run_dirs:
            run_dir = os.path.join(scenario_dir, run_dir_name)
            agents_file = os.path.join(run_dir, f"{base_filename}_agents.csv")
            rounds_file = os.path.join(run_dir, f"{base_filename}_rounds.csv")

            if os.path.exists(agents_file) and os.path.exists(rounds_file):
                try:
                    agents_df = pd.read_csv(agents_file)
                    rounds_df = pd.read_csv(rounds_file)
                    all_agents.append(agents_df)
                    all_rounds.append(rounds_df)
                except Exception as e:
                    logging.warning(f"Error reading files from {run_dir}: {e}")

        if not all_agents or not all_rounds:
            raise FileNotFoundError(
                f"No valid run data found for scenario: {scenario_name}"
            )

        agents_df = pd.concat(all_agents, ignore_index=True)
        rounds_df = pd.concat(all_rounds, ignore_index=True)
        return agents_df, rounds_df


def get_cooperation_rates(rounds_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cooperation rates by round."""
    # Add cooperation column (1 for cooperate, 0 for defect)
    rounds_df["cooperation"] = (rounds_df["move"] == "cooperate").astype(int)

    # Group by round and calculate cooperation rate
    coop_rates = rounds_df.groupby("round")["cooperation"].mean().reset_index()
    coop_rates.rename(columns={"cooperation": "cooperation_rate"}, inplace=True)

    return coop_rates


def get_strategy_cooperation_rates(rounds_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cooperation rates by round and strategy."""
    # Add cooperation column (1 for cooperate, 0 for defect)
    rounds_df["cooperation"] = (rounds_df["move"] == "cooperate").astype(int)

    # Group by round and strategy, calculate cooperation rate
    coop_rates = (
        rounds_df.groupby(["round", "strategy"])["cooperation"].mean().reset_index()
    )
    coop_rates.rename(columns={"cooperation": "cooperation_rate"}, inplace=True)

    return coop_rates


def load_network_structure(
    scenario_name: str,
    results_dir: str = "results",
    base_filename: str = "experiment_results",
    run_number: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[nx.Graph, Dict]:
    """Load network structure for a scenario.

    Args:
        scenario_name: Name of the scenario
        results_dir: Directory containing result files
        base_filename: Base filename prefix for result files
        run_number: Run number to load network from
        logger: Logger object for error reporting

    Returns:
        Tuple of (NetworkX graph, network data dictionary)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Load network structure from run directory
    scenario_dir = os.path.join(results_dir, scenario_name)
    if not os.path.isdir(scenario_dir):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Scenario directory not found: {scenario_dir}")
        return nx.Graph(), {}

    run_dir = os.path.join(scenario_dir, f"run_{run_number:02d}")
    if not os.path.isdir(run_dir):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Run directory not found: {run_dir}")
        return nx.Graph(), {}

    network_file = os.path.join(run_dir, f"{base_filename}_network.json")
    if not os.path.exists(network_file):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Network file not found: {network_file}")
        return nx.Graph(), {}

    try:
        with open(network_file, "r") as f:
            network_data = json.load(f)
        G = nx.Graph()
        G.add_nodes_from(network_data.get("nodes", []))
        G.add_edges_from(network_data.get("edges", []))
        return G, network_data
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing network JSON file: {e}")
        return nx.Graph(), {}
    except Exception as e:
        logger.error(f"Error loading network structure: {e}")
        return nx.Graph(), {}


def get_available_runs(scenario_name: str, results_dir: str = "results") -> List[int]:
    """Get list of available run numbers for a scenario.

    Args:
        scenario_name: Name of the scenario
        results_dir: Directory containing result files

    Returns:
        List of run numbers available for the scenario
    """
    scenario_dir = os.path.join(results_dir, scenario_name)
    if not os.path.isdir(scenario_dir):
        logging.warning(f"Scenario directory not found: {scenario_dir}")
        return []

    run_dirs = [d for d in os.listdir(scenario_dir) if d.startswith("run_")]
    if not run_dirs:
        logging.warning(f"No run directories found in scenario: {scenario_name}")
        return []

    runs = []
    for run_dir in run_dirs:
        try:
            run_num = int(run_dir.replace("run_", ""))
            runs.append(run_num)
        except ValueError:
            logging.warning(f"Invalid run directory name: {run_dir}")
            continue

    return sorted(runs)
