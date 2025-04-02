"""
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

# Add the parent directory to the path to import from npdl modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from npdl.core.agents import Agent
from npdl.core.environment import Environment
from npdl.core.utils import create_payoff_matrix


@pytest.fixture(scope="session")
def setup_logging():
    """Set up logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger("test_logger")


@pytest.fixture
def simple_agents():
    """Create a list of simple agents with different strategies."""
    return [
        Agent(agent_id=0, strategy="always_cooperate"),
        Agent(agent_id=1, strategy="always_defect"),
        Agent(agent_id=2, strategy="tit_for_tat"),
        Agent(agent_id=3, strategy="q_learning", learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
    ]


@pytest.fixture
def simple_environment(simple_agents):
    """Create a simple environment with a fully connected network."""
    payoff_matrix = create_payoff_matrix(len(simple_agents))
    return Environment(simple_agents, payoff_matrix, "fully_connected", {})


@pytest.fixture
def seed():
    """Set a fixed seed for tests that use randomness."""
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    return random_seed


@pytest.fixture
def test_scenario():
    """Create a simple test scenario."""
    return {
        "scenario_name": "test_scenario",
        "num_agents": 5,
        "num_rounds": 10,
        "network_type": "fully_connected",
        "network_params": {},
        "agent_strategies": {"always_cooperate": 3, "always_defect": 2},
        "payoff_type": "linear",
        "payoff_params": {"R": 3, "S": 0, "T": 5, "P": 1},
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "epsilon": 0.1,
        "logging_interval": 5
    }
