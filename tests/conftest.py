""""
Pytest configuration file for N-Person Prisoner's Dilemma simulation tests.
This file contains shared fixtures and configuration for tests.
"""
import os
import sys
import random
import pytest
import logging
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path # Use pathlib for temporary paths

# Add the parent directory to the path to import from npdl modules
# Ensure this path is correct relative to where pytest is run (usually project root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix

# --- Basic Setup Fixtures ---

@pytest.fixture(scope="session", autouse=True) # Auto-use ensures logging is set early
def setup_test_logging():
    """Set up basic logging for the test session."""
    logging.basicConfig(
        level=logging.DEBUG, # Use DEBUG during testing for more info
        format='%(asctime)s - %(levelname)-8s - %(name)-15s - %(message)s',
    )
    # Optionally silence overly verbose libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # Return a logger instance if needed in tests
    return logging.getLogger("pytest")

@pytest.fixture
def seed():
    """Set a fixed seed for tests that use randomness."""
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    # print(f"\nSEED SET TO {random_seed}\n") # Uncomment for debugging seed issues
    return random_seed

@pytest.fixture
def tmp_results_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for saving test results."""
    results_dir = tmp_path / "test_results"
    results_dir.mkdir()
    return results_dir

# --- Agent Fixtures ---

@pytest.fixture
def default_params():
    """Default parameters for agent creation."""
    return {
        "memory_length": 10,
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "epsilon": 0.1,
        "state_type": "proportion_discretized",
        "q_init_type": "zero",
        "max_possible_payoff": 5.0,
        "generosity": 0.1,
        "initial_move": "cooperate",
        "prob_coop": 0.5,
        "increase_rate": 0.1,
        "decrease_rate": 0.05,
        "beta": 0.01,
        "alpha_win": 0.05,
        "alpha_lose": 0.2,
        "alpha_avg": 0.01,
        "exploration_constant": 2.0
    }

@pytest.fixture
def basic_agents(default_params):
    """A list of agents with basic, non-learning strategies."""
    return [
        Agent(agent_id=0, strategy="always_cooperate", **default_params),
        Agent(agent_id=1, strategy="always_defect", **default_params),
        Agent(agent_id=2, strategy="tit_for_tat", **default_params),
        Agent(agent_id=3, strategy="pavlov", **default_params),
        Agent(agent_id=4, strategy="random", **default_params),
    ]

@pytest.fixture
def ql_agent(default_params):
    """A single standard Q-Learning agent."""
    return Agent(agent_id=10, strategy="q_learning", **default_params)

@pytest.fixture
def hysq_agent(default_params):
    """A single Hysteretic Q-Learning agent."""
    return Agent(agent_id=11, strategy="hysteretic_q", **default_params)

@pytest.fixture
def wolf_agent(default_params):
    """A single Wolf-PHC agent."""
    return Agent(agent_id=12, strategy="wolf_phc", **default_params)

@pytest.fixture
def lraq_agent(default_params):
    """A single LRA-Q agent."""
    return Agent(agent_id=13, strategy="lra_q", **default_params)

@pytest.fixture
def ucb1_agent(default_params):
    """A single UCB1-Q agent."""
    return Agent(agent_id=14, strategy="ucb1_q", **default_params)

@pytest.fixture
def tf2t_agent(default_params):
    """A single Tit-for-Two-Tats agent."""
    return Agent(agent_id=15, strategy="tit_for_two_tats", **default_params)


# --- Environment Fixtures ---

@pytest.fixture
def default_payoff_matrix():
    """Default linear payoff matrix for N=4"""
    return create_payoff_matrix(N=4, payoff_type="linear")

@pytest.fixture
def small_test_env(default_payoff_matrix, basic_agents, ql_agent, seed, setup_test_logging):
    """A small environment with a mix of basic and one QL agent."""
    agents = basic_agents[:3] + [ql_agent] # AC, AD, TFT, QL
    return Environment(agents, default_payoff_matrix, "fully_connected", {}, logger=setup_test_logging)

@pytest.fixture
@pytest.mark.parametrize("network_type, network_params", [
    ("fully_connected", {}),
    ("small_world", {"k": 4, "beta": 0.3}),
    ("scale_free", {"m": 2}),
    ("random", {"probability": 0.4}),
    ("regular", {"k": 4}),
])
def diverse_env(network_type, network_params, default_payoff_matrix, basic_agents, ql_agent, seed, setup_test_logging):
    """Fixture providing environments with different network types."""
    num_agents = 10
    agents = [Agent(agent_id=i, strategy=random.choice(["always_cooperate", "always_defect", "tit_for_tat", "q_learning"])) for i in range(num_agents)]
    payoff_matrix = create_payoff_matrix(num_agents)
    return Environment(agents, payoff_matrix, network_type, network_params, logger=setup_test_logging)


# --- Scenario Fixtures ---

@pytest.fixture
def base_scenario_dict():
    """A base dictionary resembling enhanced_scenarios.json entries."""
    return {
        "scenario_name": "BaseTestScenario",
        "num_agents": 10,
        "num_rounds": 50,
        "network_type": "small_world",
        "network_params": {"k": 4, "beta": 0.3},
        "agent_strategies": { "q_learning": 5, "tit_for_tat": 5 },
        "payoff_type": "linear",
        "payoff_params": {"R": 3, "S": 0, "T": 5, "P": 1},
        "state_type": "proportion_discretized",
        "q_init_type": "zero",
        "memory_length": 10,
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "epsilon": 0.1,
        "logging_interval": 51 # Disable logging during test runs
        # Add other common params here
    }

@pytest.fixture
def hysteretic_scenario_dict(base_scenario_dict):
    """Scenario focused on Hysteretic Q-Learning."""
    scenario = base_scenario_dict.copy()
    scenario.update({
        "scenario_name": "HysQTestScenario",
        "agent_strategies": { "hysteretic_q": 5, "always_defect": 5 },
        "beta": 0.01
    })
    return scenario