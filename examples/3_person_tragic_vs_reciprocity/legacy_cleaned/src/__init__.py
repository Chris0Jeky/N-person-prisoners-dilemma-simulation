"""
N-Person Prisoner's Dilemma Simulation Framework

A comprehensive framework for simulating and analyzing cooperation dynamics
in multi-agent prisoner's dilemma games.
"""

from .agents import (
    BaseAgent,
    StaticAgent,
    SimpleQLearningAgent,
    NPDLQLearningAgent,
    create_agent
)

from .game_environments import (
    PairwiseGame,
    NPersonGame,
    run_pairwise_experiment,
    run_nperson_experiment
)

from .experiment_runner import (
    ExperimentRunner,
    run_qlearning_experiments
)

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = [
    "BaseAgent",
    "StaticAgent", 
    "SimpleQLearningAgent",
    "NPDLQLearningAgent",
    "create_agent",
    "PairwiseGame",
    "NPersonGame",
    "run_pairwise_experiment",
    "run_nperson_experiment",
    "ExperimentRunner",
    "run_qlearning_experiments"
]